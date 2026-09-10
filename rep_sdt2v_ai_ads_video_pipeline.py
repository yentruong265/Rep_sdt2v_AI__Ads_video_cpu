from __future__ import annotations

"""
Zizen Labs — Rep_sdt2v_AI__Ads_video_cpu.
CPU orchestrator only:
User text -> GPT-4o-mini planner -> fal.ai Veo 3.1 Lite T2V at a duration-specific source length -> motion-interpolated high-quality extension -> optional TTS -> local mp4.

Default duration rules (source duration can be overridden per tier from RunPod ENV):
- 8s output: 4s source, at least 5 beats.
- 20s output: 4s source, at least 7 beats.
- 30s output: 4s source, at least 8 beats.
- 60s output: 4s source, at least 8 beats.
- fal.ai uses Veo 3.1 Lite with generate_audio=false: Eco=720p, Premium=1080p; optional narration audio is generated separately and muxed locally.
"""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI

# Video provider defaults for this flow. Tier model/resolution/source duration can be changed from RunPod ENV.
FAL_T2V_MODEL_ID_DEFAULT = "fal-ai/veo3.1/lite"
FAL_API_BASE_DEFAULT = "https://queue.fal.run"
FAL_SOURCE_DURATION_SEC_DEFAULT = 4
OPENAI_PLANNER_MODEL_DEFAULT = "gpt-4o-mini"
OPENAI_TTS_MODEL_DEFAULT = "gpt-4o-mini-tts"

ALLOWED_ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
FAL_VEO_LITE_ASPECT_RATIOS = {"16:9", "9:16"}
TARGET_DURATION_RULES = {
    "8": {"source_duration": 4, "minimum_beats": 5, "narration_limit": 120},
    "20": {"source_duration": 4, "minimum_beats": 7, "narration_limit": 300},
    "30": {"source_duration": 4, "minimum_beats": 8, "narration_limit": 450},
    "60": {"source_duration": 4, "minimum_beats": 8, "narration_limit": 900},
}
ALLOWED_TARGET_DURATIONS = set(TARGET_DURATION_RULES)
VIDEO_QUALITY_RESOLUTIONS = {"eco": "720p", "premium": "1080p"}
EXPECTED_VIDEO_DIMENSIONS = {
    "720p": {
        "16:9": (1280, 720), "9:16": (720, 1280),
    },
    "1080p": {
        "16:9": (1920, 1080), "9:16": (1080, 1920),
    },
}
GPT_TTS_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar"}

HUMAN_QUALITY_BLOCK = (
    "sharp faces, clear eyes, natural skin texture, correct anatomy, detailed hair, stable character identity, "
    "stable body proportions, natural micro-expressions. No blurry faces, no warped faces, no melting skin, "
    "no fused faces, no extra fingers, no missing limbs, no distorted anatomy, no duplicate people, "
    "no watermark, no flickering, no AI artifacts."
)

MANDATORY_NATURAL_ACTION_SENTENCE = (
    "All actions and movements in the video must look natural, physically plausible, temporally coherent, "
    "and free from AI errors such as warping, flicker, glitches, distorted anatomy, or unnatural motion."
)

MANDATORY_4K_QUALITY_SENTENCE = (
    "The video must have razor-sharp 4K image quality with crisp fine details, clean focus, realistic textures, "
    "stable clarity, and professional cinematic rendering. The entire video must look exceptionally lifelike, "
    "vivid, natural, and convincingly real, with realistic motion, lighting, depth, materials, and physical behavior."
)
MANDATORY_PRODUCT_AD_SENTENCE = "Create a professional product advertising video."
PLANNER_PROMPT_MAX_CHARS = 1500
PLANNER_MAX_ATTEMPTS = 3
PLANNER_BEAT_MAX_CHARS = 72


def log(msg: str) -> None:
    print(f"[Rep_sdt2v_AI__Ads_video_cpu] {msg}", flush=True)


def _str(v: Any, default: str = "") -> str:
    return str(v if v is not None else default).strip()


def normalize_aspect_ratio(value: Any) -> str:
    value = _str(value, "9:16")
    return value if value in ALLOWED_ASPECT_RATIOS else "9:16"


def _quality_token(value: Any) -> str:
    raw = re.sub(r"\s+", " ", _str(value)).strip().lower()
    if not raw:
        return ""
    if "premium" in raw or "cao cấp" in raw or "cao cap" in raw or "1080" in raw:
        return "premium"
    if "eco" in raw or "tiết kiệm" in raw or "tiet kiem" in raw or "720" in raw:
        return "eco"
    return ""


def normalize_video_quality(job: Dict[str, Any]) -> str:
    """
    Resolve the user's quality choice without depending on one fragile field.

    Explicit quality fields have priority over resolution fields because some
    workers add a default resolution even when the frontend already sent the
    user's Eco/Premium selection. Conflicting explicit quality fields are
    rejected instead of silently producing the wrong tier.
    """
    explicit_keys = (
        "video_quality", "quality", "quality_mode", "selected_quality",
        "selected_video_quality", "videoQuality",
    )
    resolution_keys = ("resolution", "video_resolution", "output_resolution")

    explicit_values = [
        (key, _str(job.get(key)), _quality_token(job.get(key)))
        for key in explicit_keys
        if _str(job.get(key))
    ]
    recognized_explicit = [(key, raw, token) for key, raw, token in explicit_values if token]
    explicit_tokens = {token for _, _, token in recognized_explicit}
    if len(explicit_tokens) > 1:
        raise ValueError(f"Conflicting video quality fields: {recognized_explicit}")

    resolution_values = [
        (key, _str(job.get(key)), _quality_token(job.get(key)))
        for key in resolution_keys
        if _str(job.get(key))
    ]
    recognized_resolution = [(key, raw, token) for key, raw, token in resolution_values if token]
    resolution_tokens = {token for _, _, token in recognized_resolution}
    if len(resolution_tokens) > 1:
        raise ValueError(f"Conflicting video resolution fields: {recognized_resolution}")

    if explicit_tokens:
        selected = next(iter(explicit_tokens))
        if resolution_tokens and selected not in resolution_tokens:
            log(
                "QUALITY CONFLICT | explicit quality wins over resolution default | "
                f"explicit={recognized_explicit} | resolution={recognized_resolution}"
            )
        return selected

    if resolution_tokens:
        return next(iter(resolution_tokens))

    supplied = explicit_values + resolution_values
    if supplied:
        raise ValueError(
            "Unsupported video quality value. Use Eco/720p or Premium/1080p. "
            f"Received: {[(key, raw) for key, raw, _ in supplied]}"
        )
    raise ValueError(
        "Missing video quality. Frontend/worker must send video_quality=eco|premium "
        "or resolution=720p|1080p."
    )


def get_video_resolution(video_quality: str) -> str:
    quality = _str(video_quality).lower()
    if quality not in VIDEO_QUALITY_RESOLUTIONS:
        raise ValueError(f"Unsupported normalized video quality: {video_quality}")
    return VIDEO_QUALITY_RESOLUTIONS[quality]


def normalize_target_duration(value: Any) -> str:
    raw = _str(value).lower()
    if not raw:
        raise ValueError("Missing target duration. Allowed values: 8s, 20s, 30s, 60s.")
    raw = raw.replace("seconds", "").replace("second", "").replace("giây", "").replace("s", "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw not in ALLOWED_TARGET_DURATIONS:
        raise ValueError(
            f"Unsupported target duration '{value}'. Allowed values: 8s, 20s, 30s, 60s."
        )
    return raw


def extract_target_duration(job: Dict[str, Any]) -> str:
    keys = (
        "target_duration_sec", "target_duration", "duration", "duration_sec",
        "output_duration", "selected_duration",
    )
    supplied = [(key, job.get(key)) for key in keys if _str(job.get(key))]
    if not supplied:
        raise ValueError(
            "Missing target duration. Frontend/worker must send one of: "
            "target_duration_sec, target_duration, duration, or duration_sec."
        )

    normalized = [(key, raw, normalize_target_duration(raw)) for key, raw in supplied]
    values = {value for _, _, value in normalized}
    if len(values) > 1:
        raise ValueError(f"Conflicting target duration fields: {normalized}")
    return normalized[0][2]


def get_duration_rule(target_duration: str) -> Dict[str, int]:
    key = normalize_target_duration(target_duration)
    rule = TARGET_DURATION_RULES.get(key)
    if not rule:
        raise ValueError(f"No duration rule configured for {target_duration}")
    return dict(rule)


def normalize_narration(job: Dict[str, Any], limit: int) -> str:
    raw = _str(job.get("narration_text") or job.get("voiceover_text") or job.get("narration") or job.get("script_text"))
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > int(limit):
        raise ValueError(f"Narration exceeds {limit} characters for the selected duration.")
    return raw


def normalize_user_prompt(job: Dict[str, Any]) -> str:
    raw = _str(job.get("prompt") or job.get("user_prompt") or job.get("story_text") or job.get("text") or job.get("content"))
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:1500]


def _clip_prompt_base(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", _str(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "), clipped.rfind(" "))
    if boundary >= max_chars // 2:
        clipped = clipped[:boundary].rstrip(" ,;:")
    return clipped


def ensure_mandatory_prompt_sentences(prompt: str) -> str:
    base = re.sub(r"\s+", " ", _str(prompt)).strip()
    mandatory_sentences = (
        MANDATORY_PRODUCT_AD_SENTENCE,
        MANDATORY_NATURAL_ACTION_SENTENCE,
        MANDATORY_4K_QUALITY_SENTENCE,
    )
    for sentence in mandatory_sentences:
        base = base.replace(sentence, " ")
    base = re.sub(r"\s+", " ", base).strip(" ,;:")
    mandatory_suffix = " ".join(mandatory_sentences)
    available = PLANNER_PROMPT_MAX_CHARS - len(mandatory_suffix) - 1
    if available < 0:
        raise RuntimeError("Mandatory planner sentences exceed the configured prompt limit.")
    clipped_base = _clip_prompt_base(base, available)
    final_prompt = f"{clipped_base} {mandatory_suffix}".strip() if clipped_base else mandatory_suffix
    if len(final_prompt) > PLANNER_PROMPT_MAX_CHARS:
        raise RuntimeError("Final planner prompt exceeds the configured character limit.")
    return final_prompt


def _json_loads_loose(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def _normalize_planner_beats(value: Any, expected_count: int) -> List[str]:
    if not isinstance(value, list):
        return []
    beats: List[str] = []
    for item in value:
        beat = re.sub(r"\s+", " ", _str(item)).strip(" .;:-")
        beat = re.sub(r"^(?:beat|shot|step)\s*\d+\s*[:.)-]?\s*", "", beat, flags=re.I)
        beat = _clip_prompt_base(beat, PLANNER_BEAT_MAX_CHARS)
        if beat:
            beats.append(beat)
    return beats if len(beats) == int(expected_count) else []


def _fallback_beats(minimum_beats: int) -> List[str]:
    middle = [
        "Establish the subject and setting in a clean, stable hero composition",
        "Begin a subtle camera move while the main subject remains sharp",
        "Reveal an important material or visual detail through natural light",
        "Show a physically plausible interaction or environmental response",
        "Shift to a complementary angle while preserving identity and proportions",
        "Add restrained cinematic motion that emphasizes depth and texture",
        "Let the action settle naturally with stable lighting and realistic physics",
    ]
    count = max(1, int(minimum_beats))
    selected = middle[:max(0, count - 1)]
    while len(selected) < count - 1:
        selected.append("Continue the coherent action with subtle realistic motion")
    selected.append("Finish on a polished final hero frame, fully visible and in focus")
    return [_clip_prompt_base(item, PLANNER_BEAT_MAX_CHARS) for item in selected[:count]]


def compose_prompt_with_verified_beats(base_prompt: str, beats: List[str]) -> str:
    normalized_beats = _normalize_planner_beats(beats, len(beats))
    if not normalized_beats:
        raise ValueError("Cannot compose prompt without a valid non-empty beat list.")

    base = re.sub(r"\s+", " ", _str(base_prompt)).strip()
    for sentence in (
        MANDATORY_PRODUCT_AD_SENTENCE,
        MANDATORY_NATURAL_ACTION_SENTENCE,
        MANDATORY_4K_QUALITY_SENTENCE,
    ):
        base = base.replace(sentence, " ")
    base = re.sub(r"\b(?:Beat|Shot|Step)\s*\d+\s*:\s*[^.]+\.?", " ", base, flags=re.I)
    base = re.sub(r"\bAction sequence\s*:\s*", " ", base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip(" ,;:")

    beat_block = "Action sequence: " + " ".join(
        f"Beat {index}: {beat}." for index, beat in enumerate(normalized_beats, 1)
    )
    mandatory_suffix = " ".join((
        MANDATORY_PRODUCT_AD_SENTENCE,
        MANDATORY_NATURAL_ACTION_SENTENCE,
        MANDATORY_4K_QUALITY_SENTENCE,
    ))
    available = PLANNER_PROMPT_MAX_CHARS - len(beat_block) - len(mandatory_suffix) - 2
    if available < 0:
        raise RuntimeError("Verified beat sequence and mandatory sentences exceed prompt limit.")

    clipped_base = _clip_prompt_base(base, available)
    parts = [part for part in (clipped_base, beat_block, mandatory_suffix) if part]
    final_prompt = " ".join(parts).strip()
    marker_count = len(re.findall(r"\bBeat\s+\d+\s*:", final_prompt))
    if marker_count != len(normalized_beats):
        raise RuntimeError(
            f"Final prompt beat validation failed: expected={len(normalized_beats)}, found={marker_count}."
        )
    if len(final_prompt) > PLANNER_PROMPT_MAX_CHARS:
        raise RuntimeError("Final planner prompt exceeds the configured character limit.")
    return final_prompt


def build_planner_instruction(
    user_prompt: str,
    aspect_ratio: str,
    target_duration: str,
    source_duration: int,
    minimum_beats: int,
    resolution: str,
    retry_reason: str = "",
) -> str:
    correction = (
        f"\nPrevious output was rejected because: {retry_reason}\n"
        "Correct that error and return a completely valid JSON object."
        if retry_reason else ""
    )
    return f"""
You are Zizen Labs' professional short AI video prompt planner for fal.ai Veo 3.1 Lite Text-to-Video.

Task:
- Read ONLY the user's text input below.
- Rewrite it into one concise, high-quality English video generation plan.
- The source clip is exactly {source_duration} seconds at {resolution}, with no native audio.
- Return EXACTLY {minimum_beats} distinct visual/action beats in the "beats" array: not fewer and not more.
- Each beat must be one short, concrete, visible action or camera/environment change.
- Keep one coherent subject and one coherent scene. Beats must form one continuous chronological action.
- Keep the user's intent, subject, mood, style, setting, and requested actions.
- Prefer small, physically plausible movements that can fit naturally inside {source_duration} seconds.
- Add cinematic camera language, realistic lighting, temporal consistency, and artifact prevention.
- The "replicate_prompt" should describe the overall scene and style. Do not number beats inside it; the pipeline appends and validates the numbered beat sequence.
- Do not include markdown. Output JSON only.{correction}

User input, max 1500 chars:
{user_prompt}

Selected settings:
- aspect_ratio: {aspect_ratio}
- source_generation_duration: {source_duration}s
- user_target_duration_after_postprocess: {target_duration}s
- exact_beat_count: {minimum_beats}
- resolution: {resolution}
- generate_audio: false

Human quality requirements to integrate compactly when relevant:
{HUMAN_QUALITY_BLOCK}

Return JSON exactly:
{{
  "replicate_prompt": "overall scene, style, camera, lighting, realism and artifact-prevention prompt",
  "beats": [
    "visible action beat 1",
    "visible action beat 2"
  ]
}}

The beats array shown above is only structural. Your actual array must contain exactly {minimum_beats} items.
""".strip()


def fallback_prompt(user_prompt: str, source_duration: int, minimum_beats: int) -> str:
    base = user_prompt or "A professional cinematic short AI video with realistic motion and emotional visual storytelling."
    overview = (
        f"{base}. Professional cinematic {source_duration}-second source video. "
        "One coherent scene, realistic natural motion, purposeful restrained camera movement, "
        "stable composition, sharp details, natural lighting, clean depth of field, and a premium commercial look. "
        "If human faces appear: sharp realistic face, crisp eyes, natural skin texture, correct anatomy, detailed hair. "
        "No blurry faces, warped faces, distorted anatomy, extra fingers, missing limbs, duplicate people, watermark, flicker, or AI artifacts. No audio."
    )
    return compose_prompt_with_verified_beats(overview, _fallback_beats(minimum_beats))


def plan_replicate_prompt(
    job: Dict[str, Any],
    user_prompt: str,
    aspect_ratio: str,
    target_duration: str,
    source_duration: int,
    minimum_beats: int,
    resolution: str,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        beats = _fallback_beats(minimum_beats)
        log("OPENAI_API_KEY missing; using deterministic verified-beat fallback prompt.")
        return {
            "replicate_prompt": fallback_prompt(user_prompt, source_duration, minimum_beats),
            "planner_model": "fallback",
            "minimum_beats": minimum_beats,
            "beats": beats,
            "beat_count": len(beats),
            "planner_attempts": 0,
            "beat_validation_passed": True,
        }

    model = os.getenv("OPENAI_PLANNER_MODEL", OPENAI_PLANNER_MODEL_DEFAULT)
    client = OpenAI()
    retry_reason = ""
    last_data: Dict[str, Any] = {}

    for attempt in range(1, PLANNER_MAX_ATTEMPTS + 1):
        instruction = build_planner_instruction(
            user_prompt, aspect_ratio, target_duration, source_duration,
            minimum_beats, resolution, retry_reason
        )
        log(
            f"Calling OpenAI planner: {model} | attempt={attempt}/{PLANNER_MAX_ATTEMPTS} | "
            f"source={source_duration}s | exact_beats={minimum_beats} | resolution={resolution}"
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0.15,
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. The beats array must contain exactly the requested "
                        "number of short visible action beats."
                    ),
                },
                {"role": "user", "content": instruction},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = _json_loads_loose(content)
        last_data = data
        beats = _normalize_planner_beats(data.get("beats"), minimum_beats)
        base_prompt = _str(data.get("replicate_prompt") or data.get("seedance_prompt"))

        errors = []
        if not base_prompt:
            errors.append("replicate_prompt is missing or empty")
        if not beats:
            actual = len(data.get("beats")) if isinstance(data.get("beats"), list) else 0
            errors.append(f"beats must contain exactly {minimum_beats} valid items; received {actual}")

        if not errors:
            prompt = compose_prompt_with_verified_beats(base_prompt, beats)
            return {
                "replicate_prompt": prompt,
                "planner_model": model,
                "raw_planner": data,
                "minimum_beats": minimum_beats,
                "beats": beats,
                "beat_count": len(beats),
                "planner_attempts": attempt,
                "beat_validation_passed": True,
            }

        retry_reason = "; ".join(errors)
        log(f"Planner output rejected | attempt={attempt} | reason={retry_reason}")

    beats = _fallback_beats(minimum_beats)
    log(
        "Planner failed beat validation after all attempts; using deterministic "
        f"verified-beat fallback. Last output={str(last_data)[:600]}"
    )
    return {
        "replicate_prompt": fallback_prompt(user_prompt, source_duration, minimum_beats),
        "planner_model": f"{model}:fallback",
        "raw_planner": last_data,
        "minimum_beats": minimum_beats,
        "beats": beats,
        "beat_count": len(beats),
        "planner_attempts": PLANNER_MAX_ATTEMPTS,
        "beat_validation_passed": True,
        "fallback_reason": retry_reason,
    }


def first_video_url(result: Any) -> str:
    if isinstance(result, str) and result.startswith("http"):
        return result
    if isinstance(result, list):
        for item in result:
            url = first_video_url(item)
            if url:
                return url
    if isinstance(result, dict):
        for key in ["video", "output", "file", "url"]:
            item = result.get(key)
            if isinstance(item, dict) and str(item.get("url", "")).startswith("http"):
                return str(item["url"])
            if isinstance(item, str) and item.startswith("http"):
                return item
        for key in ["videos", "outputs"]:
            items = result.get(key)
            if isinstance(items, list):
                for item in items:
                    url = first_video_url(item)
                    if url:
                        return url
        stack = list(result.values())
        while stack:
            item = stack.pop(0)
            if isinstance(item, dict):
                if str(item.get("url", "")).startswith("http"):
                    return str(item["url"])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, str) and item.startswith("http"):
                return item
    return ""


def _tier_env_prefix(video_quality: str) -> str:
    quality = _str(video_quality).lower()
    if quality == "eco":
        return "ADS_ECO"
    if quality == "premium":
        return "ADS_PREMIUM"
    raise ValueError(f"Unsupported normalized video quality: {video_quality}")


def get_tier_model_id(video_quality: str) -> str:
    prefix = _tier_env_prefix(video_quality)
    return _str(os.getenv(f"{prefix}_MODEL_ID"), FAL_T2V_MODEL_ID_DEFAULT)


def get_tier_resolution(video_quality: str) -> str:
    prefix = _tier_env_prefix(video_quality)
    default = VIDEO_QUALITY_RESOLUTIONS[video_quality]
    resolution = _str(os.getenv(f"{prefix}_RESOLUTION"), default).lower()
    if resolution not in {"720p", "1080p"}:
        raise ValueError(
            f"Unsupported {video_quality} resolution '{resolution}' for fal.ai Veo 3.1 Lite. "
            "Allowed values: 720p, 1080p."
        )
    return resolution


def get_tier_source_duration(video_quality: str, fallback_source_duration: int) -> int:
    prefix = _tier_env_prefix(video_quality)
    raw = _str(os.getenv(f"{prefix}_SOURCE_DURATION_SEC"), str(fallback_source_duration))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(fallback_source_duration)
    if value not in {4, 6, 8}:
        raise ValueError(
            f"Unsupported {video_quality} source duration '{value}' for fal.ai Veo 3.1 Lite. "
            "Allowed values: 4, 6, 8 seconds."
        )
    return value


def build_fal_input(
    prompt: str,
    aspect_ratio: str,
    source_duration: int,
    resolution: str,
) -> Dict[str, Any]:
    """Build a cost-locked fal.ai Veo input. Native model audio stays disabled by design."""
    if resolution not in {"720p", "1080p"}:
        raise ValueError(f"Unsupported fal.ai Veo 3.1 Lite resolution: {resolution}")
    if aspect_ratio not in FAL_VEO_LITE_ASPECT_RATIOS:
        raise ValueError(
            f"fal.ai Veo 3.1 Lite supports only aspect_ratio 16:9 or 9:16; received {aspect_ratio}."
        )
    if int(source_duration) not in {4, 6, 8}:
        raise ValueError(
            f"fal.ai Veo 3.1 Lite supports only 4s, 6s, or 8s source duration; received {source_duration}s."
        )

    input_payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration": f"{int(source_duration)}s",
        "generate_audio": False,
    }
    extra_json = os.getenv("FAL_EXTRA_INPUT_JSON", "").strip()
    if extra_json:
        try:
            extra = json.loads(extra_json)
            if isinstance(extra, dict):
                input_payload.update(extra)
        except Exception as e:
            log(f"Ignoring invalid FAL_EXTRA_INPUT_JSON: {e}")

    # These fields are hard-locked after optional extras so tier/cost/audio rules cannot be overridden.
    input_payload["duration"] = f"{int(source_duration)}s"
    input_payload["resolution"] = resolution
    input_payload["generate_audio"] = False
    input_payload["aspect_ratio"] = aspect_ratio
    log(
        f"FAL FINAL INPUT | duration={input_payload.get('duration')} | "
        f"resolution={input_payload.get('resolution')} | generate_audio={input_payload.get('generate_audio')} | "
        f"aspect_ratio={input_payload.get('aspect_ratio')}"
    )
    return input_payload


def call_fal_t2v(
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    source_duration: int,
    resolution: str,
) -> Dict[str, Any]:
    token = _str(os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY"))
    if not token:
        raise RuntimeError("Missing FAL_KEY (or FAL_API_KEY) environment variable.")

    model_id = _str(model_id, FAL_T2V_MODEL_ID_DEFAULT).strip("/")
    if "/" not in model_id:
        raise RuntimeError("fal.ai model id must look like owner/model, for example fal-ai/veo3.1/lite")

    base = _str(os.getenv("FAL_API_BASE"), FAL_API_BASE_DEFAULT).rstrip("/")
    submit_url = f"{base}/{model_id}"
    arguments = build_fal_input(prompt, aspect_ratio, source_duration, resolution)
    headers = {"Authorization": f"Key {token}", "Content-Type": "application/json"}

    log(
        f"Calling fal.ai {model_id} | source_duration={source_duration}s | "
        f"resolution={resolution} | aspect_ratio={aspect_ratio} | generate_audio=false"
    )
    start = time.time()
    request_timeout = _bounded_env_int("FAL_REQUEST_TIMEOUT_SEC", 120, 30, 600)
    res = requests.post(submit_url, headers=headers, json=arguments, timeout=request_timeout)
    text = res.text
    try:
        queued = res.json()
    except Exception:
        queued = {"raw": text}
    if not res.ok:
        raise RuntimeError(f"fal.ai submit failed: status={res.status_code}, body={text[:1200]}")

    request_id = _str(queued.get("request_id") or queued.get("requestId"))
    status_url = _str(queued.get("status_url") or queued.get("statusUrl"))
    response_url = _str(queued.get("response_url") or queued.get("responseUrl"))
    if not request_id:
        # Some compatible endpoints can return the final payload synchronously.
        video_url = first_video_url(queued.get("video")) or first_video_url(queued)
        if video_url:
            elapsed = time.time() - start
            result_payload = queued
            return {
                "fal_video_url": video_url,
                "fal_raw_result": result_payload,
                "fal_elapsed_sec": elapsed,
                "fal_model": model_id,
                "fal_input": arguments,
                # Backward-compatible aliases used by the current handler/output contract.
                "replicate_video_url": video_url,
                "replicate_raw_result": result_payload,
                "replicate_elapsed_sec": elapsed,
                "replicate_model": model_id,
                "replicate_input": arguments,
            }
        raise RuntimeError(f"fal.ai submit response missing request_id: {str(queued)[:1200]}")

    if not status_url or not response_url:
        # Queue API normally returns both URLs; construct standard fal queue URLs as a defensive fallback.
        request_base = f"{base}/{model_id}/requests/{request_id}"
        status_url = status_url or f"{request_base}/status"
        response_url = response_url or f"{request_base}/response"

    deadline = time.time() + _bounded_env_int("FAL_POLL_TIMEOUT_SEC", 900, 60, 3600)
    poll_interval = _bounded_env_int("FAL_POLL_INTERVAL_SEC", 5, 1, 60)
    status = "IN_QUEUE"
    last_status_payload: Dict[str, Any] = queued
    failure_statuses = {"FAILED", "ERROR", "CANCELLED", "CANCELED", "TIMED_OUT"}

    while status != "COMPLETED":
        if time.time() > deadline:
            raise TimeoutError(f"fal.ai prediction timed out. request_id={request_id}, last_status={status}")
        time.sleep(poll_interval)
        poll = requests.get(status_url, headers={"Authorization": f"Key {token}"}, timeout=60)
        poll_text = poll.text
        try:
            last_status_payload = poll.json()
        except Exception:
            last_status_payload = {"raw": poll_text}
        if not poll.ok:
            raise RuntimeError(f"fal.ai status poll failed: status={poll.status_code}, body={poll_text[:1200]}")
        status = _str(last_status_payload.get("status")).upper()
        log(f"fal.ai status={status or 'UNKNOWN'} request_id={request_id}")
        if status in failure_statuses:
            raise RuntimeError(
                f"fal.ai prediction ended with status={status}, error="
                f"{last_status_payload.get('error') or last_status_payload.get('detail') or last_status_payload}"
            )
        if not status:
            raise RuntimeError(f"fal.ai status response missing status: {str(last_status_payload)[:1200]}")

    # fal queue can report COMPLETED while including an error/error_type field.
    # Do not attempt to fetch or treat such a request as a successful video generation.
    completed_error = last_status_payload.get("error") or last_status_payload.get("error_type")
    if completed_error:
        raise RuntimeError(
            f"fal.ai prediction completed with error: request_id={request_id}, "
            f"error={completed_error}"
        )

    result_res = requests.get(response_url, headers={"Authorization": f"Key {token}"}, timeout=120)
    result_text = result_res.text
    try:
        result_payload = result_res.json()
    except Exception:
        result_payload = {"raw": result_text}
    if not result_res.ok:
        raise RuntimeError(f"fal.ai result fetch failed: status={result_res.status_code}, body={result_text[:1200]}")

    elapsed = time.time() - start
    video_url = first_video_url(result_payload.get("video")) or first_video_url(result_payload)
    if not video_url:
        raise RuntimeError(f"Could not find video URL in fal.ai result: {str(result_payload)[:1200]}")
    return {
        "fal_video_url": video_url,
        "fal_raw_result": result_payload,
        "fal_elapsed_sec": elapsed,
        "fal_model": model_id,
        "fal_input": arguments,
        # Backward-compatible aliases used by the current handler/output contract.
        "replicate_video_url": video_url,
        "replicate_raw_result": result_payload,
        "replicate_elapsed_sec": elapsed,
        "replicate_model": model_id,
        "replicate_input": arguments,
    }


# Backward-compatible function name for any caller that still imports the old symbol.
def call_replicate_t2v(
    prompt: str,
    aspect_ratio: str,
    source_duration: int,
    resolution: str,
    model_id: str | None = None,
) -> Dict[str, Any]:
    return call_fal_t2v(
        model_id=model_id or FAL_T2V_MODEL_ID_DEFAULT,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        source_duration=source_duration,
        resolution=resolution,
    )

def download_video(video_url: str, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading video: {video_url[:140]}")
    with requests.get(video_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    size = Path(out_path).stat().st_size
    if size < 2048:
        raise RuntimeError(f"Downloaded video too small ({size} bytes)")
    return out_path


def _run(cmd: List[str], timeout_sec: int | None = None) -> subprocess.CompletedProcess:
    log("Running: " + " ".join(cmd))
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_sec}s: {' '.join(cmd)}"
        ) from exc
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT={p.stdout[-1000:]}\nSTDERR={p.stderr[-2000:]}")
    return p


def probe_duration_sec(path: str) -> float:
    p = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    try:
        return max(float(p.stdout.strip()), 0.1)
    except Exception:
        return float(FAL_SOURCE_DURATION_SEC_DEFAULT)


def probe_video_dimensions(path: str) -> Dict[str, int]:
    p = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", path,
    ])
    try:
        payload = json.loads(p.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except Exception as exc:
        raise RuntimeError(f"Could not parse video dimensions for {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {path}: {width}x{height}")
    return {"width": width, "height": height}


def validate_actual_resolution(
    path: str, requested_resolution: str, aspect_ratio: str, stage: str
) -> Dict[str, int]:
    """Validate the actual configured video tier using expected dimensions for each ratio.

    Use a ratio-aware expected-dimension table so 720p and 1080p outputs are
    validated against the requested orientation without relying on one generic
    short-side threshold.
    """
    dimensions = probe_video_dimensions(path)
    expected_by_ratio = EXPECTED_VIDEO_DIMENSIONS.get(requested_resolution)
    expected = expected_by_ratio.get(aspect_ratio) if expected_by_ratio else None
    if not expected:
        raise RuntimeError(
            f"No expected dimensions configured for resolution={requested_resolution}, "
            f"aspect_ratio={aspect_ratio}."
        )

    expected_width, expected_height = expected
    actual_width = dimensions["width"]
    actual_height = dimensions["height"]
    tolerance = 0.90
    width_ok = actual_width >= int(expected_width * tolerance)
    height_ok = actual_height >= int(expected_height * tolerance)
    if not (width_ok and height_ok):
        raise RuntimeError(
            f"{stage} resolution validation failed: requested={requested_resolution} "
            f"at {aspect_ratio}, expected about {expected_width}x{expected_height}, "
            f"actual={actual_width}x{actual_height}."
        )

    log(
        f"{stage} RESOLUTION VERIFIED | requested={requested_resolution} | "
        f"aspect_ratio={aspect_ratio} | expected={expected_width}x{expected_height} | "
        f"actual={actual_width}x{actual_height}"
    )
    return dimensions


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_str(os.getenv(name), str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(_str(os.getenv(name), str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_flag(name: str, default: bool = True) -> bool:
    raw = _str(os.getenv(name), "1" if default else "0").lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


_EXECUTOR_INIT_LOCK = threading.Lock()
_TTS_EXECUTOR: ThreadPoolExecutor | None = None
_POSTPROCESS_EXECUTOR: ThreadPoolExecutor | None = None


def _get_tts_executor() -> ThreadPoolExecutor:
    """Dedicated TTS pool so job threads can safely wait without executor starvation."""
    global _TTS_EXECUTOR
    if _TTS_EXECUTOR is None:
        with _EXECUTOR_INIT_LOCK:
            if _TTS_EXECUTOR is None:
                default_workers = _bounded_env_int("RUNPOD_WORKER_CONCURRENCY", 1, 1, 32)
                workers = _bounded_env_int("TTS_MAX_CONCURRENCY", default_workers, 1, 32)
                _TTS_EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ads-tts")
                log(f"TTS executor initialized | max_workers={workers}")
    return _TTS_EXECUTOR


def _get_postprocess_executor() -> ThreadPoolExecutor:
    """Bound heavy FFmpeg work independently from RunPod request concurrency."""
    global _POSTPROCESS_EXECUTOR
    if _POSTPROCESS_EXECUTOR is None:
        with _EXECUTOR_INIT_LOCK:
            if _POSTPROCESS_EXECUTOR is None:
                workers = _bounded_env_int("POSTPROCESS_MAX_CONCURRENCY", 1, 1, 8)
                _POSTPROCESS_EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ads-ffmpeg")
                log(f"Postprocess executor initialized | max_workers={workers}")
    return _POSTPROCESS_EXECUTOR


def _raise_if_future_failed(future: Future | None, checkpoint: str) -> None:
    """Fail fast if the parallel audio branch has already ended with an error."""
    if future is None or not future.done():
        return
    exc = future.exception()
    if exc is not None:
        raise RuntimeError(f"Parallel narration failed before {checkpoint}: {exc}") from exc


def _build_brand_drawtext_filter(brand_name: str, job_id: str) -> str:
    """Build the same brand overlay as before, but for the main postprocess pass."""
    brand = re.sub(r"[\r\n\t]+", " ", _str(brand_name)).strip()[:60]
    if not brand:
        return ""

    configured_font = _str(os.getenv("ADS_BRAND_FONT_FILE"))
    if configured_font and not Path(configured_font).exists():
        raise RuntimeError(
            f"ADS_BRAND_FONT_FILE points to a missing file: {configured_font}. "
            "Remove the ENV to use automatic font discovery or set it to an existing Unicode font."
        )
    font_candidates = [
        configured_font,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    font_file = next((candidate for candidate in font_candidates if candidate and Path(candidate).exists()), "")
    if not font_file:
        try:
            match = subprocess.run(
                ["fc-match", "-f", "%{file}", "DejaVu Sans:style=Bold"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
            )
            discovered = _str(match.stdout)
            if match.returncode == 0 and discovered and Path(discovered).exists():
                font_file = discovered
        except Exception:
            font_file = ""

    text_file = f"/tmp/{job_id}_brand.txt"
    Path(text_file).write_text(brand, encoding="utf-8")
    font_color = _str(os.getenv("ADS_BRAND_FONT_COLOR"), "white@0.94")
    font_ratio = float(os.getenv("ADS_BRAND_FONT_SIZE_RATIO", "0.026"))
    margin_ratio = float(os.getenv("ADS_BRAND_MARGIN_RATIO", "0.035"))
    font_arg = f"fontfile='{font_file}'" if font_file else "font='Sans'"
    return (
        f"drawtext={font_arg}:textfile='{text_file}':reload=0:"
        f"fontcolor={font_color}:fontsize=h*{font_ratio:.4f}:"
        f"x=w*{margin_ratio:.4f}:y=h-th-h*{margin_ratio:.4f}:"
        "shadowcolor=black@0.72:shadowx=2:shadowy=2"
    )


def extend_video_to_duration(
    input_path: str,
    output_path: str,
    target_duration_sec: int,
    brand_name: str = "",
    job_id: str = "",
) -> Dict[str, Any]:
    """
    Extend a short source clip without asking optical flow to invent a very
    large number of frames after the clip has already been slowed down.

    Anti-warp strategy:
    1. Interpolate the ORIGINAL adjacent frames to a bounded internal FPS.
    2. Slow the already-denser timeline with setpts.
    3. Convert to the requested output FPS using frame selection/duplication.
    4. Encode at low CRF, with sharpening disabled or kept extremely mild.

    This order is intentionally different from the previous implementation.
    Running setpts first and minterpolate afterward forces motion estimation
    across an artificially enlarged temporal gap, which can create liquid,
    wavy edges around products, food, hands, faces, steam, and reflections.

    If minterpolate is unavailable or fails, the function falls back to a
    distortion-free frame-duplication slow-motion path. The fallback can look
    less fluid, but it will not create optical-flow warping.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    target_duration_sec = int(target_duration_sec)
    source_duration = probe_duration_sec(input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ratio = target_duration_sec / max(source_duration, 0.1)
    output_fps = _bounded_env_int("POSTPROCESS_FPS", 24, 18, 60)
    max_interpolation_fps = _bounded_env_int("POSTPROCESS_MAX_INTERPOLATION_FPS", 72, output_fps, 120)
    crf = _bounded_env_int("POSTPROCESS_CRF", 12, 0, 28)
    sharpen_amount = _bounded_env_float("POSTPROCESS_SHARPEN_AMOUNT", 0.10, 0.0, 0.35)
    use_motion_interpolation = _env_flag("POSTPROCESS_USE_MOTION_INTERPOLATION", True)
    preset = _str(os.getenv("POSTPROCESS_X264_PRESET"), "slow").lower()
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        preset = "slow"
    ffmpeg_threads = _bounded_env_int("POSTPROCESS_FFMPEG_THREADS", 0, 0, 256)
    brand_filter = _build_brand_drawtext_filter(brand_name, job_id) if brand_name else ""

    # Interpolate only between the ORIGINAL neighboring frames. The internal
    # frame rate is bounded so CPU usage and optical-flow synthesis stay sane.
    desired_internal_fps = int(round(output_fps * min(max(ratio, 1.0), 3.0)))
    internal_fps = max(output_fps, min(desired_internal_fps, max_interpolation_fps))

    optional_sharpen = ""
    if sharpen_amount > 0.001:
        # Very restrained luma-only sharpening. Strong sharpening makes AI
        # textures and interpolation errors look more synthetic.
        optional_sharpen = f",unsharp=5:5:{sharpen_amount:.2f}:3:3:0.0"

    brand_suffix = f",{brand_filter}" if brand_filter else ""
    common_tail = (
        f"setpts={ratio:.8f}*PTS,"
        f"fps={output_fps}:round=near"
        f"{optional_sharpen}"
        f"{brand_suffix},"
        "tpad=stop_mode=clone:stop_duration=2"
    )

    anti_warp_filter = (
        "format=yuv420p,setpts=PTS-STARTPTS,"
        f"minterpolate=fps={internal_fps}:mi_mode=mci:mc_mode=obmc:"
        "me_mode=bilat:me=epzs:mb_size=16:search_param=32:"
        "vsbmc=0:scd=fdiff:scd_threshold=8,"
        f"{common_tail}"
    )

    # No motion-compensated synthesis in the fallback. It only retimes and
    # duplicates existing frames, so it cannot bend straight lines or create
    # the liquid/wave artifact seen in aggressive optical flow.
    no_warp_fallback_filter = (
        "format=yuv420p,setpts=PTS-STARTPTS,"
        f"{common_tail}"
    )

    def encode_with_filter(filter_chain: str) -> None:
        cmd = ["ffmpeg", "-y"]
        if ffmpeg_threads > 0:
            cmd.extend(["-filter_threads", str(ffmpeg_threads)])
        cmd.extend([
            "-i", input_path,
            "-map", "0:v:0",
            "-filter:v", filter_chain,
            "-t", str(target_duration_sec),
            "-an",
            "-c:v", "libx264",
        ])
        if ffmpeg_threads > 0:
            cmd.extend(["-threads", str(ffmpeg_threads)])
        cmd.extend([
            "-preset", preset,
            "-crf", str(crf),
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr",
            "-movflags", "+faststart",
            output_path,
        ])
        postprocess_timeout = _bounded_env_int("POSTPROCESS_TIMEOUT_SEC", 600, 60, 3600)
        _run(cmd, timeout_sec=postprocess_timeout)

    requested_strategy = "anti_warp_preinterpolated_slowmo" if use_motion_interpolation else "no_warp_frame_duplication_slowmo"
    strategy = requested_strategy
    interpolation_fallback_reason = ""

    log(
        f"POSTPROCESS START | source_path={input_path} | output_path={output_path} | "
        f"source_duration={source_duration:.3f}s | target_duration={target_duration_sec}s | "
        f"ratio={ratio:.4f} | output_fps={output_fps} | internal_fps={internal_fps} | "
        f"crf={crf} | sharpen={sharpen_amount:.2f} | strategy={requested_strategy} | "
        f"brand_overlay={bool(brand_filter)} | ffmpeg_threads={ffmpeg_threads or 'auto'}"
    )

    if use_motion_interpolation:
        try:
            encode_with_filter(anti_warp_filter)
        except RuntimeError as exc:
            interpolation_fallback_reason = str(exc)[-900:]
            strategy = "no_warp_frame_duplication_fallback"
            log(
                "Anti-warp interpolation failed; retrying without optical flow. "
                f"Reason: {interpolation_fallback_reason}"
            )
            encode_with_filter(no_warp_fallback_filter)
    else:
        encode_with_filter(no_warp_fallback_filter)

    if not Path(output_path).exists() or Path(output_path).stat().st_size < 2048:
        raise RuntimeError(f"Postprocess failed: output file not created or too small: {output_path}")

    final_duration = probe_duration_sec(output_path)
    if final_duration < target_duration_sec - 0.45:
        raise RuntimeError(
            f"Postprocess duration validation failed: target={target_duration_sec}s, final={final_duration:.3f}s."
        )

    log(
        f"POSTPROCESS DONE | strategy={strategy} | final_duration={final_duration:.3f}s | "
        f"output_fps={output_fps} | internal_fps={internal_fps} | crf={crf}"
    )
    return {
        "extended": True,
        "source_duration_sec": source_duration,
        "target_duration_sec": target_duration_sec,
        "final_duration_sec": final_duration,
        "strategy": strategy,
        "slowmo_ratio": ratio,
        "motion_interpolation_requested": use_motion_interpolation,
        "motion_interpolation_applied": strategy == "anti_warp_preinterpolated_slowmo",
        "interpolation_order": "before_timestamp_stretch",
        "interpolation_fallback_reason": interpolation_fallback_reason,
        "output_fps": output_fps,
        "internal_interpolation_fps": internal_fps,
        "x264_crf": crf,
        "x264_preset": preset,
        "ffmpeg_threads": ffmpeg_threads,
        "brand_overlay_applied": bool(brand_filter),
        "sharpen_amount": sharpen_amount,
        "anti_warp_settings": {
            "mc_mode": "obmc",
            "me_mode": "bilat",
            "vsbmc": 0,
            "scene_change_detection": "fdiff",
            "scene_change_threshold": 8,
        },
    }

def normalize_gpt_voice(selection: str) -> str:
    value = _str(selection or "shimmer").lower()
    if value.startswith("gpt:"):
        value = value.split(":", 1)[1]
    if value.startswith("other_languages_"):
        value = value.replace("other_languages_", "", 1)
    return value if value in GPT_TTS_VOICES else "shimmer"


def create_openai_tts(text: str, out_path: str, voice: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY for GPT TTS")
    client = OpenAI()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=os.getenv("OPENAI_TTS_MODEL", OPENAI_TTS_MODEL_DEFAULT),
        voice=normalize_gpt_voice(voice),
        input=text,
    ) as response:
        response.stream_to_file(out_path)
    if not Path(out_path).exists() or Path(out_path).stat().st_size < 512:
        raise RuntimeError("OpenAI TTS output too small")
    return out_path


def normalize_runpod_run_url(url: str) -> str:
    clean = _str(url).rstrip("/")
    if clean.endswith("/run") or clean.endswith("/runsync"):
        return clean
    return clean + "/run" if clean else ""


def get_runpod_status_url(run_url: str, request_id: str) -> str:
    clean = _str(run_url).rstrip("/")
    if clean.endswith("/run"):
        clean = clean[:-4]
    elif clean.endswith("/runsync"):
        clean = clean[:-8]
    return f"{clean}/status/{request_id}"


def extract_omnivoice_audio_output(obj: Any) -> Dict[str, Any]:
    data = obj if isinstance(obj, dict) else {}
    output = data.get("output") if isinstance(data.get("output"), dict) else data
    status = _str(data.get("status") or output.get("status")).upper()
    audio_url = _str(
        output.get("audio_url")
        or output.get("result_audio_url")
        or output.get("output_url")
        or output.get("r2_url")
        or output.get("url")
        or data.get("audio_url")
        or data.get("result_audio_url")
    )
    if audio_url and not audio_url.lower().startswith(("http://", "https://")):
        audio_url = ""
    error_message = _str(
        output.get("error_message")
        or output.get("error")
        or data.get("error_message")
        or data.get("error")
    )
    return {"status": status, "audio_url": audio_url, "error_message": error_message, "output": output}


def probe_audio_duration_sec(path: str) -> float:
    p = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration:format=duration", "-of", "json", path,
    ])
    try:
        payload = json.loads(p.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("no audio stream")
        stream_duration = streams[0].get("duration")
        format_duration = (payload.get("format") or {}).get("duration")
        raw = stream_duration if stream_duration not in (None, "", "N/A") else format_duration
        duration = float(raw)
    except Exception as exc:
        raise RuntimeError(f"Could not determine audio duration for {path}: {exc}") from exc
    if duration <= 0:
        raise RuntimeError(f"Invalid audio duration for {path}: {duration}")
    return duration


def download_audio(audio_url: str, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(audio_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(out_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
    if not Path(out_path).exists() or Path(out_path).stat().st_size < 512:
        raise RuntimeError("Downloaded TTS audio is empty or too small")
    return out_path


def call_omnivoice_tts(text: str, out_path: str, job: Dict[str, Any], cancel_event: threading.Event | None = None) -> str:
    url = _str(os.getenv("RUNPOD_OMNIVOICE_ENDPOINT_URL") or os.getenv("RUNPOD_OMNIVOICE_API_URL"))
    key = _str(os.getenv("RUNPOD_OMNIVOICE_API_KEY") or os.getenv("RUNPOD_API_KEY"))
    if not url or not key:
        raise RuntimeError("Missing RUNPOD_OMNIVOICE_ENDPOINT_URL/API_KEY for VN or clone voice TTS")
    profile = job.get("voice_profile") or {}
    if not isinstance(profile, dict) or not profile.get("ref_audio_url"):
        raise RuntimeError("Missing voice_profile.ref_audio_url for OmniVoice TTS")
    audio_job_id = f"audio_{_str(job.get('job_id'), uuid.uuid4().hex)}_{uuid.uuid4().hex[:8]}"
    payload = {
        "input": {
            "job_id": audio_job_id,
            "text": text,
            "prompt": text,
            "ref_audio_url": profile.get("ref_audio_url"),
            "ref_text": profile.get("ref_text") or text[:180],
            "voice_id": profile.get("voice_id") or job.get("voice"),
            "language": profile.get("language") or "Vietnamese",
            "language_id": profile.get("language_id") or "vi",
            "language_iso3": profile.get("language_iso3") or "vie",
            "speed": float(os.getenv("OMNIVOICE_SPEED", "1.0")),
            "num_step": int(os.getenv("OMNIVOICE_NUM_STEP", "32")),
        }
    }
    run_url = normalize_runpod_run_url(url)
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OmniVoice cancelled before submission because the video branch failed.")
    tts_started = time.perf_counter()
    log(f"OMNIVOICE SUBMIT | parent_job={job.get('job_id')} | audio_job={audio_job_id}")
    response = requests.post(run_url, headers=headers, json=payload, timeout=120)
    text_response = response.text
    try:
        data = response.json()
    except Exception:
        data = {"raw": text_response}
    if not response.ok:
        raise RuntimeError(f"OmniVoice request failed: status={response.status_code}, body={text_response[:1000]}")

    request_id = data.get("id") or data.get("request_id") or data.get("runpod_request_id")
    extracted = extract_omnivoice_audio_output(data)
    status = extracted["status"]
    audio_url = extracted["audio_url"]
    success_statuses = {"COMPLETED", "SUCCESS", "SUCCEEDED"}
    failure_statuses = {"FAILED", "CANCELLED", "CANCELED", "TIMED_OUT"}

    # /run is asynchronous. An output URL may become visible before the generating handler
    # has actually finished, so only an explicit terminal-success status plus a generated
    # audio URL is accepted as complete.
    if request_id:
        status_url = get_runpod_status_url(run_url, str(request_id))
        poll_timeout = int(os.getenv("OMNIVOICE_POLL_TIMEOUT_SEC", "900"))
        poll_interval = max(1, int(os.getenv("OMNIVOICE_POLL_INTERVAL_SEC", "4")))
        log_interval = _bounded_env_int("OMNIVOICE_STATUS_LOG_INTERVAL_SEC", 15, 5, 120)
        deadline = time.time() + poll_timeout
        last_logged_status = ""
        last_log_at = 0.0
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("OmniVoice polling cancelled because the video branch failed.")
            if extracted["error_message"] or status in failure_statuses:
                raise RuntimeError(
                    f"OmniVoice failed: status={status}, error={extracted['error_message']}, response={str(data)[:1000]}"
                )
            now = time.time()
            if status != last_logged_status or now - last_log_at >= log_interval:
                log(
                    f"OMNIVOICE STATUS | parent_job={job.get('job_id')} | request_id={request_id} | "
                    f"status={status or 'UNKNOWN'} | elapsed={time.perf_counter() - tts_started:.1f}s"
                )
                last_logged_status = status
                last_log_at = now
            if status in success_statuses:
                if not audio_url:
                    raise RuntimeError(
                        f"OmniVoice reached terminal success without a generated audio URL: request_id={request_id}"
                    )
                break
            time.sleep(poll_interval)
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("OmniVoice polling cancelled because the video branch failed.")
            poll = requests.get(status_url, headers={"authorization": f"Bearer {key}"}, timeout=60)
            poll_text = poll.text
            try:
                data = poll.json()
            except Exception:
                data = {"raw": poll_text}
            if not poll.ok:
                raise RuntimeError(f"OmniVoice poll failed: status={poll.status_code}, body={poll_text[:1000]}")
            extracted = extract_omnivoice_audio_output(data)
            status = extracted["status"]
            audio_url = extracted["audio_url"]
        else:
            raise RuntimeError(f"OmniVoice timed out before validated completion: request_id={request_id}, last_status={status}")
    else:
        if extracted["error_message"] or status not in success_statuses or not audio_url:
            raise RuntimeError(
                "OmniVoice response has no request id and is not a validated synchronous completion: "
                f"status={status}, audio_url={bool(audio_url)}, error={extracted['error_message']}"
            )

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OmniVoice cancelled before audio download because the video branch failed.")
    download_audio(audio_url, out_path)
    if cancel_event is not None and cancel_event.is_set():
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("OmniVoice cancelled after audio download because the video branch failed.")
    audio_duration = probe_audio_duration_sec(out_path)
    if audio_duration <= float(os.getenv("OMNIVOICE_MIN_VALID_AUDIO_SEC", "0.5")):
        raise RuntimeError(f"OmniVoice completed audio is too short: {audio_duration:.3f}s")
    log(
        f"OMNIVOICE AUDIO COMPLETE | parent_job={job.get('job_id')} | request_id={request_id or 'sync'} | "
        f"duration={audio_duration:.3f}s | elapsed={time.perf_counter() - tts_started:.1f}s"
    )
    return out_path


def create_narration_tts(
    text: str,
    out_path: str,
    job: Dict[str, Any],
    cancel_event: threading.Event | None = None,
) -> str:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Narration cancelled before generation because the video branch failed.")
    provider = _str(job.get("voice_provider")).lower()
    selection = _str(job.get("voice_selection") or job.get("voice") or job.get("primary_voice") or job.get("tts_voice") or "shimmer")
    profile = job.get("voice_profile")
    if provider in {"flozen_public_omnivoice", "flozen_clone_omnivoice", "omnivoice"} or isinstance(profile, dict):
        return call_omnivoice_tts(text, out_path, job, cancel_event=cancel_event)
    result = create_openai_tts(text, out_path, selection)
    if cancel_event is not None and cancel_event.is_set():
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("Narration cancelled after generation because the video branch failed.")
    return result


def mux_video_audio(video_path: str, audio_path: str, output_path: str, duration_sec: float) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0", "-t", f"{float(duration_sec):.3f}",
        "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", output_path,
    ])
    return output_path


def normalize_brand_name(job: Dict[str, Any]) -> str:
    raw = _str(job.get("brand_name") or job.get("channel_name") or job.get("brand_text"))
    return re.sub(r"\s+", " ", raw).strip()[:60]


def apply_brand_overlay(input_path: str, output_path: str, brand_name: str, job_id: str) -> str:
    brand = re.sub(r"[\r\n\t]+", " ", _str(brand_name)).strip()[:60]
    if not brand:
        if input_path != output_path:
            Path(output_path).write_bytes(Path(input_path).read_bytes())
        return output_path

    configured_font = _str(os.getenv("ADS_BRAND_FONT_FILE"))
    if configured_font and not Path(configured_font).exists():
        raise RuntimeError(
            f"ADS_BRAND_FONT_FILE points to a missing file: {configured_font}. "
            "Remove the ENV to use automatic font discovery or set it to an existing Unicode font."
        )
    font_candidates = [
        configured_font,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    font_file = next((candidate for candidate in font_candidates if candidate and Path(candidate).exists()), "")
    if not font_file:
        try:
            match = subprocess.run(
                ["fc-match", "-f", "%{file}", "DejaVu Sans:style=Bold"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
            )
            discovered = _str(match.stdout)
            if match.returncode == 0 and discovered and Path(discovered).exists():
                font_file = discovered
        except Exception:
            font_file = ""
    text_file = f"/tmp/{job_id}_brand.txt"
    Path(text_file).write_text(brand, encoding="utf-8")
    font_color = _str(os.getenv("ADS_BRAND_FONT_COLOR"), "white@0.94")
    font_ratio = float(os.getenv("ADS_BRAND_FONT_SIZE_RATIO", "0.026"))
    margin_ratio = float(os.getenv("ADS_BRAND_MARGIN_RATIO", "0.035"))
    crf = str(_bounded_env_int("POSTPROCESS_CRF", 12, 0, 28))
    preset = _str(os.getenv("POSTPROCESS_X264_PRESET"), "slow").lower()
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        preset = "slow"
    font_arg = f"fontfile='{font_file}'" if font_file else "font='Sans'"
    drawtext = (
        f"drawtext={font_arg}:textfile='{text_file}':reload=0:"
        f"fontcolor={font_color}:fontsize=h*{font_ratio:.4f}:"
        f"x=w*{margin_ratio:.4f}:y=h-th-h*{margin_ratio:.4f}:"
        "shadowcolor=black@0.72:shadowx=2:shadowy=2"
    )
    _run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", drawtext,
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-c:a", "copy", "-movflags", "+faststart", output_path,
    ])
    if not Path(output_path).exists() or Path(output_path).stat().st_size < 2048:
        raise RuntimeError("Brand overlay output is missing or too small")
    log(f"BRAND OVERLAY DONE | brand={brand!r} | position=bottom-left | font={font_file or 'fontconfig:Sans'}")
    return output_path


def generate_rep_sdt2v_ai_ads_video(job: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_started = time.perf_counter()
    job_id = _str(job.get("job_id"), uuid.uuid4().hex)
    user_prompt = normalize_user_prompt(job)
    if not user_prompt:
        raise ValueError("Missing prompt. Please enter a text request up to 1500 characters.")

    aspect_ratio = normalize_aspect_ratio(job.get("aspect_ratio"))
    video_quality = normalize_video_quality(job)
    resolution = get_tier_resolution(video_quality)
    model_id = get_tier_model_id(video_quality)
    target_duration = extract_target_duration(job)
    target_duration_sec = int(target_duration)
    duration_rule = get_duration_rule(target_duration)
    source_duration = get_tier_source_duration(video_quality, int(duration_rule["source_duration"]))
    minimum_beats = int(duration_rule["minimum_beats"])
    narration_limit = int(duration_rule["narration_limit"])
    narration = normalize_narration(job, narration_limit)
    generate_audio = bool(narration)
    brand_name = normalize_brand_name(job)

    raw_path = f"/tmp/{job_id}_fal_veo31lite_{source_duration}s_source.mp4"
    silent_path = f"/tmp/{job_id}_rep_sdt2v_ai_ads_video_silent.mp4"
    final_path = f"/tmp/{job_id}_rep_sdt2v_ai_ads_video_final.mp4"
    audio_path = f"/tmp/{job_id}_narration.mp3"

    quality_inputs = {
        key: job.get(key) for key in (
            "video_quality", "quality", "quality_mode", "selected_quality",
            "selected_video_quality", "videoQuality", "resolution",
            "video_resolution", "output_resolution",
        ) if _str(job.get(key))
    }
    duration_inputs = {
        key: job.get(key) for key in (
            "target_duration_sec", "target_duration", "duration", "duration_sec",
            "output_duration", "selected_duration",
        ) if _str(job.get(key))
    }
    log(
        f"REQUEST SETTINGS | target={target_duration_sec}s | source={source_duration}s | "
        f"minimum_beats={minimum_beats} | narration_limit={narration_limit} | "
        f"aspect_ratio={aspect_ratio} | video_quality={video_quality} | model_id={model_id} | resolution={resolution} | "
        f"generate_audio={generate_audio} | brand_name={brand_name!r} | quality_inputs={quality_inputs} | duration_inputs={duration_inputs}"
    )

    audio_future: Future | None = None
    audio_cancel_event = threading.Event()
    stage_timings: Dict[str, float] = {}

    # Start narration immediately after request validation. It uses only narration/voice data
    # and can therefore overlap GPT planning, fal.ai generation, download and postprocessing.
    if generate_audio:
        log(f"TTS QUEUED | parent_job={job_id} | provider={job.get('voice_provider')}")
        audio_future = _get_tts_executor().submit(
            create_narration_tts, narration, audio_path, job, audio_cancel_event
        )

    try:
        planner_started = time.perf_counter()
        plan = plan_replicate_prompt(
            job, user_prompt, aspect_ratio, target_duration, source_duration, minimum_beats, resolution
        )
        stage_timings["planner"] = time.perf_counter() - planner_started
        log(f"STAGE DONE | planner | elapsed={stage_timings['planner']:.2f}s")
        _raise_if_future_failed(audio_future, "fal.ai submission")

        # Prompt construction is intentionally unchanged.
        prompt = _str(plan.get("replicate_prompt") or plan.get("seedance_prompt")) or fallback_prompt(user_prompt, source_duration, minimum_beats)
        prompt = ensure_mandatory_prompt_sentences(prompt)

        fal_started = time.perf_counter()
        rep_result = call_fal_t2v(
            model_id=model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            source_duration=source_duration,
            resolution=resolution,
        )
        stage_timings["fal"] = time.perf_counter() - fal_started
        log(f"STAGE DONE | fal | elapsed={stage_timings['fal']:.2f}s")
        _raise_if_future_failed(audio_future, "video download")

        download_started = time.perf_counter()
        download_video(rep_result["replicate_video_url"], raw_path)
        source_duration_after_download = probe_duration_sec(raw_path)
        source_dimensions = validate_actual_resolution(raw_path, resolution, aspect_ratio, "SOURCE")
        stage_timings["download_validate"] = time.perf_counter() - download_started
        log(
            f"SOURCE VIDEO READY | path={raw_path} | duration={source_duration_after_download:.3f}s | "
            f"dimensions={source_dimensions['width']}x{source_dimensions['height']} | "
            f"elapsed={stage_timings['download_validate']:.2f}s"
        )
        _raise_if_future_failed(audio_future, "postprocess")

        # Heavy FFmpeg work is queued in a dedicated pool. This lets the endpoint accept
        # multiple jobs while independently limiting CPU-heavy postprocessing concurrency.
        video_stage_path = silent_path if generate_audio else final_path
        postprocess_queued = time.perf_counter()
        log(f"POSTPROCESS QUEUED | parent_job={job_id}")
        postprocess_future = _get_postprocess_executor().submit(
            extend_video_to_duration,
            raw_path,
            video_stage_path,
            target_duration_sec,
            brand_name,
            job_id,
        )
        # While waiting for the scarce CPU slot, react immediately if the parallel
        # narration branch fails. If FFmpeg is still queued, cancel it so a failed job
        # never consumes a postprocess slot. If FFmpeg is already running, let that one
        # pass finish before raising so temp cleanup cannot race an active subprocess.
        warned_audio_failed_while_running = False
        audio_completion_checked = audio_future is None
        while not postprocess_future.done():
            watched = [postprocess_future]
            if audio_future is not None and not audio_completion_checked:
                watched.append(audio_future)
            done, _ = wait(watched, return_when=FIRST_COMPLETED)
            if audio_future is not None and audio_future in done and not audio_completion_checked:
                audio_completion_checked = True
                audio_exc = audio_future.exception()
                if audio_exc is not None:
                    if postprocess_future.cancel():
                        raise RuntimeError(
                            f"Parallel narration failed while postprocess was queued: {audio_exc}"
                        ) from audio_exc
                    if not warned_audio_failed_while_running:
                        log(
                            f"TTS FAILED WHILE POSTPROCESS RUNNING | parent_job={job_id} | "
                            "waiting for current FFmpeg pass to finish safely"
                        )
                        warned_audio_failed_while_running = True
        extend_info = postprocess_future.result()
        stage_timings["postprocess_wait_and_run"] = time.perf_counter() - postprocess_queued
        log(f"STAGE DONE | postprocess | elapsed={stage_timings['postprocess_wait_and_run']:.2f}s")
        _raise_if_future_failed(audio_future, "audio join")

        audio_duration = 0.0
        if generate_audio:
            audio_wait_started = time.perf_counter()
            if audio_future is None:
                raise RuntimeError("Narration future was not created for an audio-enabled job.")
            if not audio_future.done():
                log(f"TTS WAIT | parent_job={job_id} | video branch complete; waiting for narration")
            audio_future.result()
            stage_timings["audio_join_wait"] = time.perf_counter() - audio_wait_started

            audio_duration = probe_audio_duration_sec(audio_path)
            if audio_duration <= float(os.getenv("OMNIVOICE_MIN_VALID_AUDIO_SEC", "0.5")):
                raise RuntimeError(f"Narration audio is too short to mux: {audio_duration:.3f}s")
            overflow_tolerance = _bounded_env_float("ADS_AUDIO_MAX_OVERFLOW_SEC", 0.15, 0.0, 1.0)
            if audio_duration > float(target_duration_sec) + overflow_tolerance:
                raise RuntimeError(
                    f"Narration audio ({audio_duration:.3f}s) exceeds selected video duration ({target_duration_sec}s). "
                    "Refusing to truncate the narration; shorten the narration or choose a longer duration."
                )

            mux_started = time.perf_counter()
            # Brand is already rendered in the single main video encode. Mux therefore copies
            # the video stream and encodes only AAC audio.
            mux_video_audio(silent_path, audio_path, final_path, target_duration_sec)
            muxed_audio_duration = probe_audio_duration_sec(final_path)
            required_audio_duration = min(audio_duration, float(target_duration_sec))
            if muxed_audio_duration + 0.12 < required_audio_duration:
                raise RuntimeError(
                    f"Mux validation failed: source narration={audio_duration:.3f}s, "
                    f"muxed audio stream={muxed_audio_duration:.3f}s, target={target_duration_sec}s."
                )
            stage_timings["audio_mux"] = time.perf_counter() - mux_started
            audio_complete = True
            log(
                f"AUDIO MUX VERIFIED | voice_provider={job.get('voice_provider')} | "
                f"source_audio={audio_duration:.3f}s | muxed_audio={muxed_audio_duration:.3f}s | "
                f"mux_elapsed={stage_timings['audio_mux']:.2f}s"
            )
        else:
            audio_complete = True

        final_validate_started = time.perf_counter()
        final_duration = probe_duration_sec(final_path)
        if final_duration < target_duration_sec - 0.45:
            raise RuntimeError(f"Final video is too short: target={target_duration_sec}s, actual={final_duration:.3f}s")
        final_dimensions = validate_actual_resolution(final_path, resolution, aspect_ratio, "FINAL")
        stage_timings["final_validate"] = time.perf_counter() - final_validate_started
        stage_timings["pipeline_total"] = time.perf_counter() - pipeline_started
        log(
            f"FINAL VIDEO READY | path={final_path} | duration={final_duration:.3f}s | "
            f"dimensions={final_dimensions['width']}x{final_dimensions['height']}"
        )
        log(
            f"PIPELINE TIMING | parent_job={job_id} | total={stage_timings['pipeline_total']:.2f}s | "
            f"planner={stage_timings.get('planner', 0):.2f}s | fal={stage_timings.get('fal', 0):.2f}s | "
            f"download_validate={stage_timings.get('download_validate', 0):.2f}s | "
            f"postprocess_wait_and_run={stage_timings.get('postprocess_wait_and_run', 0):.2f}s | "
            f"audio_join_wait={stage_timings.get('audio_join_wait', 0):.2f}s | "
            f"audio_mux={stage_timings.get('audio_mux', 0):.2f}s"
        )

        return {
            "local_path": final_path,
            "source_local_path": raw_path,
            "fal_video_url": rep_result.get("fal_video_url", rep_result["replicate_video_url"]),
            "fal_model": rep_result.get("fal_model", model_id),
            "fal_input": rep_result.get("fal_input", rep_result.get("replicate_input", {})),
            "replicate_video_url": rep_result["replicate_video_url"],
            "replicate_model": rep_result.get("replicate_model", model_id),
            "replicate_input": rep_result.get("replicate_input", {}),
            "planner_model": plan.get("planner_model", os.getenv("OPENAI_PLANNER_MODEL", OPENAI_PLANNER_MODEL_DEFAULT)),
            "planner": plan,
            "prompt": prompt,
            "user_prompt": user_prompt,
            "minimum_beats": minimum_beats,
            "beats": plan.get("beats", []),
            "beat_count": int(plan.get("beat_count") or 0),
            "beat_validation_passed": bool(plan.get("beat_validation_passed")),
            "planner_attempts": int(plan.get("planner_attempts") or 0),
            "narration_text": narration,
            "narration_chars": len(narration),
            "narration_limit_chars": narration_limit,
            "voice_selection": _str(job.get("voice_selection") or job.get("voice") or job.get("primary_voice") or job.get("tts_voice")),
            "voice_provider": _str(job.get("voice_provider")),
            "audio_duration_sec": audio_duration,
            "audio_complete": bool(audio_complete),
            "brand_name": brand_name,
            "fal_elapsed_sec": rep_result.get("fal_elapsed_sec", rep_result.get("replicate_elapsed_sec", 0)),
            "replicate_elapsed_sec": rep_result.get("replicate_elapsed_sec", 0),
            "aspect_ratio": aspect_ratio,
            "video_quality": video_quality,
            "resolution": resolution,
            "source_dimensions": source_dimensions,
            "final_dimensions": final_dimensions,
            "duration": target_duration,
            "duration_sec": target_duration_sec,
            "final_duration_sec": final_duration,
            "replicate_source_duration": str(source_duration),
            "replicate_source_duration_sec": source_duration,
            "postprocess_extend": extend_info,
            "stage_timings_sec": stage_timings,
            "generate_audio": generate_audio,
            "provider": "fal_veo31lite_ai_product_ads_text_video",
            "pipeline_mode": "Rep_sdt2v_AI__Ads_video_cpu",
        }
    except Exception:
        # Best-effort stop of the local parallel audio branch if the video branch fails.
        # OmniVoice polling checks this event between polls, preventing needless local waiting/download.
        if audio_future is not None and not audio_future.done():
            audio_cancel_event.set()
            cancelled = audio_future.cancel()
            if not cancelled:
                try:
                    audio_future.result(timeout=_bounded_env_int("TTS_CANCEL_JOIN_TIMEOUT_SEC", 10, 1, 60))
                except FutureTimeoutError:
                    log(f"WARNING: TTS branch did not stop within cancellation join timeout | parent_job={job_id}")
                except Exception:
                    pass
        raise


# Compatibility aliases for callers that use a generic text-to-video function name.
generate_rep_sdt2v_ai_video = generate_rep_sdt2v_ai_ads_video
generate_ai_text_video_seedance = generate_rep_sdt2v_ai_ads_video
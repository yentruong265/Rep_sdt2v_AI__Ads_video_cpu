from __future__ import annotations

"""
Zizen Labs — Rep_sdt2v_AI__Ads_video_cpu.
CPU orchestrator only:
User text -> GPT-4o-mini planner -> Replicate Seedance 2.0 Mini T2V at a duration-specific source length -> motion-interpolated high-quality extension -> optional TTS -> local mp4.

Locked duration rules:
- 8s output: 4s source, at least 5 beats.
- 15s output: 4s source, at least 7 beats.
- 30s output: 5s source, at least 9 beats.
- 60s output: 8s source, at least 14 beats.
- Replicate is always 480p with generate_audio=false; optional narration audio is generated separately and muxed locally.
"""

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI

# Locked model for this flow: Seedance 2.0 Mini Text-to-Video on Replicate.
# Do not switch this endpoint to another model accidentally.
REPLICATE_T2V_MODEL_ID_DEFAULT = "bytedance/seedance-2.0-mini"
REPLICATE_T2V_MODEL_ID = REPLICATE_T2V_MODEL_ID_DEFAULT
OPENAI_PLANNER_MODEL_DEFAULT = "gpt-4o-mini"
OPENAI_TTS_MODEL_DEFAULT = "gpt-4o-mini-tts"

ALLOWED_ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
TARGET_DURATION_RULES = {
    "8": {"source_duration": 4, "minimum_beats": 5, "narration_limit": 120},
    "15": {"source_duration": 4, "minimum_beats": 5, "narration_limit": 225},
    "30": {"source_duration": 5, "minimum_beats": 7, "narration_limit": 450},
    "60": {"source_duration": 7, "minimum_beats": 9, "narration_limit": 900},
}
ALLOWED_TARGET_DURATIONS = set(TARGET_DURATION_RULES)
REPLICATE_SOURCE_DURATION_SEC_DEFAULT = 4
REPLICATE_API_BASE = "https://api.replicate.com/v1"
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


def log(msg: str) -> None:
    print(f"[Rep_sdt2v_AI__Ads_video_cpu] {msg}", flush=True)


def _str(v: Any, default: str = "") -> str:
    return str(v if v is not None else default).strip()


def normalize_aspect_ratio(value: Any) -> str:
    value = _str(value, "9:16")
    return value if value in ALLOWED_ASPECT_RATIOS else "9:16"


def normalize_target_duration(value: Any) -> str:
    raw = _str(value, "8").lower()
    raw = raw.replace("seconds", "").replace("second", "").replace("giây", "").replace("s", "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw if raw in ALLOWED_TARGET_DURATIONS else "8"


def get_duration_rule(target_duration: str) -> Dict[str, int]:
    return dict(TARGET_DURATION_RULES.get(str(target_duration), TARGET_DURATION_RULES["8"]))


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


def build_planner_instruction(
    user_prompt: str,
    aspect_ratio: str,
    target_duration: str,
    source_duration: int,
    minimum_beats: int,
) -> str:
    return f"""
You are FlozenAI's professional short AI video prompt planner for Replicate Seedance Text-to-Video.

Task:
- Read ONLY the user's text input below.
- Rewrite it into one concise, high-quality English video generation prompt.
- The model source clip will be exactly {source_duration} seconds at 480p with no audio.
- Keep one coherent subject and one coherent scene, but structure the motion as a continuous sequence of AT LEAST {minimum_beats} distinct visual/action beats.
- Each beat must be concrete and visible: body movement, object interaction, environmental reaction, camera movement, or a clear change in framing.
- Keep the user's intent, subject, mood, style, setting, and requested actions.
- Make the sequence physically plausible and temporally continuous so it remains stable during motion-interpolated extension to {target_duration} seconds.
- Add cinematic camera language, lighting, temporal consistency, and artifact-prevention instructions.
- If people/humans/faces are involved, integrate the human quality requirements compactly.
- The final replicate_prompt must contain these three exact mandatory sentences, each exactly once:
  1. {MANDATORY_PRODUCT_AD_SENTENCE}
  2. {MANDATORY_NATURAL_ACTION_SENTENCE}
  3. {MANDATORY_4K_QUALITY_SENTENCE}
- The complete replicate_prompt, including all three mandatory sentences, must not exceed {PLANNER_PROMPT_MAX_CHARS} characters.
- Do not include markdown. Output JSON only.

User input, max 1500 chars:
{user_prompt}

Selected settings:
- aspect_ratio: {aspect_ratio}
- replicate_generation_duration: {source_duration}s
- user_target_duration_after_postprocess: {target_duration}s
- minimum_beats: {minimum_beats}
- resolution: 480p
- generate_audio: false

Human quality requirements to integrate compactly when relevant:
{HUMAN_QUALITY_BLOCK}

Return JSON exactly:
{{
  "replicate_prompt": "final prompt here, max 1500 chars, including the three mandatory sentences exactly once"
}}
""".strip()


def fallback_prompt(user_prompt: str, source_duration: int, minimum_beats: int) -> str:
    base = user_prompt or "A professional cinematic short AI video with realistic motion and emotional visual storytelling."
    prompt = (
        f"{base}. Professional cinematic {source_duration}-second source video. "
        f"Show one coherent continuous scene with at least {minimum_beats} distinct visible action beats, smoothly connected in chronological order. "
        "Realistic natural motion, purposeful camera movement, stable composition, sharp details, natural lighting, clean depth of field, high-quality commercial look. "
        "If human faces appear: sharp realistic face, crisp eyes, natural skin texture, correct anatomy, detailed hair. "
        "No blurry faces, no warped faces, no distorted anatomy, no extra fingers, no missing limbs, no duplicate people, no watermark, no flicker, no AI artifacts. No audio."
    )
    return ensure_mandatory_prompt_sentences(prompt)


def plan_replicate_prompt(
    job: Dict[str, Any],
    user_prompt: str,
    aspect_ratio: str,
    target_duration: str,
    source_duration: int,
    minimum_beats: int,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        log("OPENAI_API_KEY missing; using fallback prompt.")
        return {
            "replicate_prompt": fallback_prompt(user_prompt, source_duration, minimum_beats),
            "planner_model": "fallback",
            "minimum_beats": minimum_beats,
        }

    model = os.getenv("OPENAI_PLANNER_MODEL", OPENAI_PLANNER_MODEL_DEFAULT)
    client = OpenAI()
    instruction = build_planner_instruction(user_prompt, aspect_ratio, target_duration, source_duration, minimum_beats)
    log(f"Calling OpenAI planner: {model} | source={source_duration}s | minimum_beats={minimum_beats}")
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You rewrite user video ideas into concise Replicate Seedance prompts with the required number of visible action beats. Output valid JSON only."},
            {"role": "user", "content": instruction},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    data = _json_loads_loose(content)
    prompt = _str(data.get("replicate_prompt") or data.get("seedance_prompt"))
    if not prompt:
        prompt = fallback_prompt(user_prompt, source_duration, minimum_beats)
    prompt = ensure_mandatory_prompt_sentences(prompt)
    return {
        "replicate_prompt": prompt,
        "planner_model": model,
        "raw_planner": data,
        "minimum_beats": minimum_beats,
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


def build_replicate_input(prompt: str, aspect_ratio: str, source_duration: int) -> Dict[str, Any]:
    """Build a cost-locked Seedance input for the selected output-duration rule."""
    input_payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": "480p",
        "duration": int(source_duration),
        "generate_audio": False,
    }
    extra_json = os.getenv("REPLICATE_EXTRA_INPUT_JSON", "").strip()
    if extra_json:
        try:
            extra = json.loads(extra_json)
            if isinstance(extra, dict):
                input_payload.update(extra)
        except Exception as e:
            log(f"Ignoring invalid REPLICATE_EXTRA_INPUT_JSON: {e}")

    # These three fields are hard-locked after optional extras so cost/audio rules cannot be overridden.
    input_payload["duration"] = int(source_duration)
    input_payload["resolution"] = "480p"
    input_payload["generate_audio"] = False
    log(
        f"REPLICATE FINAL INPUT | duration={input_payload.get('duration')} | "
        f"resolution={input_payload.get('resolution')} | generate_audio={input_payload.get('generate_audio')} | "
        f"aspect_ratio={input_payload.get('aspect_ratio')}"
    )
    return input_payload


def call_replicate_t2v(prompt: str, aspect_ratio: str, source_duration: int) -> Dict[str, Any]:
    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN environment variable.")

    model_id = REPLICATE_T2V_MODEL_ID_DEFAULT
    if "/" not in model_id:
        raise RuntimeError("Replicate model id must look like owner/model, for example bytedance/seedance-1.5-pro")
    owner, model_name = model_id.split("/", 1)
    url = f"{REPLICATE_API_BASE}/models/{owner}/{model_name}/predictions"
    arguments = build_replicate_input(prompt, aspect_ratio, source_duration)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait=60"}

    log(f"Calling Replicate {model_id} | source_duration={source_duration}s | resolution=480p | aspect_ratio={aspect_ratio}")
    start = time.time()
    res = requests.post(url, headers=headers, json={"input": arguments}, timeout=90)
    text = res.text
    try:
        pred = res.json()
    except Exception:
        pred = {"raw": text}
    if not res.ok:
        raise RuntimeError(f"Replicate create prediction failed: status={res.status_code}, body={text[:1200]}")

    status = str(pred.get("status", "")).lower()
    get_url = pred.get("urls", {}).get("get") or f"{REPLICATE_API_BASE}/predictions/{pred.get('id')}"
    deadline = time.time() + int(os.getenv("REPLICATE_POLL_TIMEOUT_SEC", "900"))
    while status not in {"succeeded", "failed", "canceled"}:
        if time.time() > deadline:
            raise TimeoutError(f"Replicate prediction timed out after polling. prediction_id={pred.get('id')}")
        time.sleep(int(os.getenv("REPLICATE_POLL_INTERVAL_SEC", "5")))
        poll = requests.get(get_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        poll_text = poll.text
        try:
            pred = poll.json()
        except Exception:
            pred = {"raw": poll_text}
        if not poll.ok:
            raise RuntimeError(f"Replicate poll failed: status={poll.status_code}, body={poll_text[:1200]}")
        status = str(pred.get("status", "")).lower()
        log(f"Replicate status={status} prediction_id={pred.get('id')}")

    elapsed = time.time() - start
    if status != "succeeded":
        raise RuntimeError(f"Replicate prediction ended with status={status}, error={pred.get('error')}")
    video_url = first_video_url(pred.get("output")) or first_video_url(pred)
    if not video_url:
        raise RuntimeError(f"Could not find video URL in Replicate result: {str(pred)[:1200]}")
    return {
        "replicate_video_url": video_url,
        "replicate_raw_result": pred,
        "replicate_elapsed_sec": elapsed,
        "replicate_model": model_id,
        "replicate_input": arguments,
    }


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


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    log("Running: " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        return float(REPLICATE_SOURCE_DURATION_SEC_DEFAULT)


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


def extend_video_to_duration(input_path: str, output_path: str, target_duration_sec: int) -> Dict[str, Any]:
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

    # Interpolate only between the ORIGINAL neighboring frames. The internal
    # frame rate is bounded so CPU usage and optical-flow synthesis stay sane.
    desired_internal_fps = int(round(output_fps * min(max(ratio, 1.0), 3.0)))
    internal_fps = max(output_fps, min(desired_internal_fps, max_interpolation_fps))

    optional_sharpen = ""
    if sharpen_amount > 0.001:
        # Very restrained luma-only sharpening. Strong sharpening makes AI
        # textures and interpolation errors look more synthetic.
        optional_sharpen = f",unsharp=5:5:{sharpen_amount:.2f}:3:3:0.0"

    common_tail = (
        f"setpts={ratio:.8f}*PTS,"
        f"fps={output_fps}:round=near"
        f"{optional_sharpen},"
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
        _run([
            "ffmpeg", "-y", "-i", input_path,
            "-map", "0:v:0",
            "-filter:v", filter_chain,
            "-t", str(target_duration_sec),
            "-an",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr",
            "-movflags", "+faststart",
            output_path,
        ])

    requested_strategy = "anti_warp_preinterpolated_slowmo" if use_motion_interpolation else "no_warp_frame_duplication_slowmo"
    strategy = requested_strategy
    interpolation_fallback_reason = ""

    log(
        f"POSTPROCESS START | source_path={input_path} | output_path={output_path} | "
        f"source_duration={source_duration:.3f}s | target_duration={target_duration_sec}s | "
        f"ratio={ratio:.4f} | output_fps={output_fps} | internal_fps={internal_fps} | "
        f"crf={crf} | sharpen={sharpen_amount:.2f} | strategy={requested_strategy}"
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


def extract_audio_url(obj: Any) -> str:
    if isinstance(obj, str) and obj.startswith("http"):
        return obj
    if isinstance(obj, list):
        for item in obj:
            url = extract_audio_url(item)
            if url:
                return url
    if isinstance(obj, dict):
        for key in ["audio_url", "result_audio_url", "output_url", "r2_url", "url"]:
            value = obj.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in obj.values():
            url = extract_audio_url(value)
            if url:
                return url
    return ""


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


def call_omnivoice_tts(text: str, out_path: str, job: Dict[str, Any]) -> str:
    url = _str(os.getenv("RUNPOD_OMNIVOICE_ENDPOINT_URL") or os.getenv("RUNPOD_OMNIVOICE_API_URL"))
    key = _str(os.getenv("RUNPOD_OMNIVOICE_API_KEY") or os.getenv("RUNPOD_API_KEY"))
    if not url or not key:
        raise RuntimeError("Missing RUNPOD_OMNIVOICE_ENDPOINT_URL/API_KEY for VN or clone voice TTS")
    profile = job.get("voice_profile") or {}
    if not isinstance(profile, dict) or not profile.get("ref_audio_url"):
        raise RuntimeError("Missing voice_profile.ref_audio_url for OmniVoice TTS")
    payload = {
        "input": {
            "job_id": f"audio_{uuid.uuid4().hex[:12]}",
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
    response = requests.post(run_url, headers=headers, json=payload, timeout=120)
    text_response = response.text
    try:
        data = response.json()
    except Exception:
        data = {"raw": text_response}
    if not response.ok:
        raise RuntimeError(f"OmniVoice request failed: status={response.status_code}, body={text_response[:1000]}")
    audio_url = extract_audio_url(data)
    request_id = data.get("id") or data.get("request_id") or data.get("runpod_request_id")
    status = str(data.get("status", "")).upper()
    if not audio_url and request_id:
        status_url = get_runpod_status_url(run_url, str(request_id))
        deadline = time.time() + int(os.getenv("OMNIVOICE_POLL_TIMEOUT_SEC", "900"))
        while time.time() < deadline:
            time.sleep(int(os.getenv("OMNIVOICE_POLL_INTERVAL_SEC", "4")))
            poll = requests.get(status_url, headers={"authorization": f"Bearer {key}"}, timeout=60)
            poll_text = poll.text
            try:
                poll_data = poll.json()
            except Exception:
                poll_data = {"raw": poll_text}
            if not poll.ok:
                raise RuntimeError(f"OmniVoice poll failed: status={poll.status_code}, body={poll_text[:1000]}")
            audio_url = extract_audio_url(poll_data)
            status = str(poll_data.get("status") or poll_data.get("output", {}).get("status") or "").upper()
            if audio_url or status in {"COMPLETED", "FAILED", "CANCELLED", "CANCELED"}:
                if status == "FAILED" and not audio_url:
                    raise RuntimeError(f"OmniVoice failed: {str(poll_data)[:1000]}")
                break
    if not audio_url:
        raise RuntimeError(f"Could not find audio URL in OmniVoice response: {str(data)[:1000]}")
    return download_audio(audio_url, out_path)


def create_narration_tts(text: str, out_path: str, job: Dict[str, Any]) -> str:
    provider = _str(job.get("voice_provider")).lower()
    selection = _str(job.get("voice_selection") or job.get("voice") or job.get("primary_voice") or job.get("tts_voice") or "shimmer")
    profile = job.get("voice_profile")
    if provider in {"flozen_public_omnivoice", "flozen_clone_omnivoice", "omnivoice"} or isinstance(profile, dict):
        return call_omnivoice_tts(text, out_path, job)
    return create_openai_tts(text, out_path, selection)


def mux_video_audio(video_path: str, audio_path: str, output_path: str, duration_sec: float) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0", "-t", f"{float(duration_sec):.3f}",
        "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", output_path,
    ])
    return output_path


def generate_rep_sdt2v_ai_ads_video(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = _str(job.get("job_id"), uuid.uuid4().hex)
    user_prompt = normalize_user_prompt(job)
    if not user_prompt:
        raise ValueError("Missing prompt. Please enter a text request up to 1500 characters.")

    aspect_ratio = normalize_aspect_ratio(job.get("aspect_ratio"))
    target_duration = normalize_target_duration(job.get("target_duration_sec") or job.get("duration") or job.get("duration_sec"))
    target_duration_sec = int(target_duration)
    duration_rule = get_duration_rule(target_duration)
    source_duration = int(duration_rule["source_duration"])
    minimum_beats = int(duration_rule["minimum_beats"])
    narration_limit = int(duration_rule["narration_limit"])
    narration = normalize_narration(job, narration_limit)
    generate_audio = bool(narration)

    log(
        f"REQUEST SETTINGS | target={target_duration_sec}s | source={source_duration}s | "
        f"minimum_beats={minimum_beats} | aspect_ratio={aspect_ratio} | generate_audio={generate_audio}"
    )
    plan = plan_replicate_prompt(job, user_prompt, aspect_ratio, target_duration, source_duration, minimum_beats)
    prompt = _str(plan.get("replicate_prompt") or plan.get("seedance_prompt")) or fallback_prompt(user_prompt, source_duration, minimum_beats)
    prompt = ensure_mandatory_prompt_sentences(prompt)

    rep_result = call_replicate_t2v(prompt=prompt, aspect_ratio=aspect_ratio, source_duration=source_duration)
    raw_path = f"/tmp/{job_id}_replicate_seedance_{source_duration}s_source.mp4"
    silent_path = f"/tmp/{job_id}_rep_sdt2v_ai_ads_video_silent.mp4"
    final_path = f"/tmp/{job_id}_rep_sdt2v_ai_ads_video_final.mp4"
    download_video(rep_result["replicate_video_url"], raw_path)
    source_duration_after_download = probe_duration_sec(raw_path)
    log(f"SOURCE VIDEO READY | path={raw_path} | duration={source_duration_after_download:.3f}s")

    extend_output_path = silent_path if generate_audio else final_path
    extend_info = extend_video_to_duration(raw_path, extend_output_path, target_duration_sec)
    audio_duration = 0.0
    if generate_audio:
        audio_path = f"/tmp/{job_id}_narration.mp3"
        create_narration_tts(narration, audio_path, job)
        audio_duration = probe_duration_sec(audio_path)
        mux_video_audio(silent_path, audio_path, final_path, target_duration_sec)
        log(f"AUDIO MUX DONE | voice_provider={job.get('voice_provider')} | audio_duration={audio_duration:.3f}s")

    final_duration = probe_duration_sec(final_path)
    if final_duration < target_duration_sec - 0.45:
        raise RuntimeError(f"Final video is too short: target={target_duration_sec}s, actual={final_duration:.3f}s")
    log(f"FINAL VIDEO READY | path={final_path} | duration={final_duration:.3f}s")

    return {
        "local_path": final_path,
        "source_local_path": raw_path,
        "replicate_video_url": rep_result["replicate_video_url"],
        "replicate_model": rep_result.get("replicate_model", REPLICATE_T2V_MODEL_ID),
        "replicate_input": rep_result.get("replicate_input", {}),
        "planner_model": plan.get("planner_model", os.getenv("OPENAI_PLANNER_MODEL", OPENAI_PLANNER_MODEL_DEFAULT)),
        "planner": plan,
        "prompt": prompt,
        "user_prompt": user_prompt,
        "minimum_beats": minimum_beats,
        "narration_text": narration,
        "narration_chars": len(narration),
        "narration_limit_chars": narration_limit,
        "voice_selection": _str(job.get("voice_selection") or job.get("voice") or job.get("primary_voice") or job.get("tts_voice")),
        "voice_provider": _str(job.get("voice_provider")),
        "audio_duration_sec": audio_duration,
        "replicate_elapsed_sec": rep_result.get("replicate_elapsed_sec", 0),
        "aspect_ratio": aspect_ratio,
        "resolution": "480p",
        "duration": target_duration,
        "duration_sec": target_duration_sec,
        "final_duration_sec": final_duration,
        "replicate_source_duration": str(source_duration),
        "replicate_source_duration_sec": source_duration,
        "postprocess_extend": extend_info,
        "generate_audio": generate_audio,
        "provider": "replicate_seedance_ai_product_ads_text_video",
        "pipeline_mode": "Rep_sdt2v_AI__Ads_video_cpu",
    }


# Compatibility aliases for callers that use a generic text-to-video function name.
generate_rep_sdt2v_ai_video = generate_rep_sdt2v_ai_ads_video
generate_ai_text_video_seedance = generate_rep_sdt2v_ai_ads_video
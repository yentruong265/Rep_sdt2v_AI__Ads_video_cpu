from __future__ import annotations

"""
RunPod Serverless handler — Rep_sdt2v_AI__Ads_video_cpu.
CPU orchestrator: GPT-4o-mini planner -> duration-specific fal.ai Veo source -> slow motion -> optional TTS mux -> R2 -> Cloudflare callback.
"""

import asyncio
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import runpod

from rep_sdt2v_ai_ads_video_pipeline import generate_rep_sdt2v_ai_ads_video, log
from r2_utils import upload_to_r2


def _send_callback(callback_url: str, token: str, payload: Dict[str, Any]) -> None:
    if not callback_url:
        return
    try:
        import requests
        headers = {"Content-Type": "application/json"}
        if token:
            headers["x-internal-token"] = token
            headers["authorization"] = f"Bearer {token}"
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=30)
        log(f"Callback sent to {callback_url}: {resp.status_code}")
    except Exception as e:
        log(f"WARNING: callback failed: {e}")


def _cleanup_job_temp_files(job_id: str) -> None:
    """Remove only temp files created by this ads job; never touch other jobs."""
    prefix = f"{job_id}_"
    removed = 0
    try:
        for path in Path("/tmp").iterdir():
            if path.is_file() and path.name.startswith(prefix):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    log(f"WARNING: temp cleanup failed for {path}: {exc}")
        if removed:
            log(f"TEMP CLEANUP DONE | job={job_id} | files_removed={removed}")
    except Exception as exc:
        log(f"WARNING: temp cleanup scan failed for job={job_id}: {exc}")


def _handle_job(event: Dict[str, Any]) -> Dict[str, Any]:
    job: Dict[str, Any] = event.get("input", {})
    job_id: str = str(job.get("job_id") or "unknown")
    has_real_job_id = bool(job.get("job_id"))
    callback_url = str(job.get("callback_url") or os.getenv("CALLBACK_URL") or "").strip()
    token = os.getenv("INTERNAL_CALLBACK_TOKEN", "")

    job_started = time.perf_counter()
    log(f"=== JOB START: {job_id} ===")
    log(f"Job type: {job.get('job_type')} | pipeline_mode: {job.get('pipeline_mode')}")

    try:
        result = generate_rep_sdt2v_ai_ads_video(job)
        local_path: str = result["local_path"]

        log(f"Pipeline done. Local path: {local_path}")

        if bool(result.get("generate_audio", False)) and (result.get("audio_complete") is not True or not (float(result.get("audio_duration_sec", 0) or 0) > 0)):
            raise RuntimeError(
                f"Refusing completed status because narration audio is not validated complete "
                f"(audio_complete={result.get('audio_complete')}, audio_duration_sec={result.get('audio_duration_sec')})."
            )

        r2_key = f"videos/{job_id}/output.mp4"
        public_url = upload_to_r2(local_path, r2_key)
        log(f"R2 upload done: {public_url}")

        output = {
            "ok": True,
            "job_id": job_id,
            "status": "completed",
            "step": "done",
            "progress_pct": 100,
            "result_video_key": r2_key,
            "result_video_url": public_url,
            "fal_video_url": result.get("fal_video_url", result.get("replicate_video_url", "")),
            "fal_model": result.get("fal_model", result.get("replicate_model", "")),
            # Backward-compatible aliases retained for existing callback consumers.
            "replicate_video_url": result.get("replicate_video_url", result.get("fal_video_url", "")),
            "replicate_model": result.get("replicate_model", result.get("fal_model", "")),
            "replicate_source_duration": result.get("replicate_source_duration", ""),
            "postprocess_extend": result.get("postprocess_extend", {}),
            "planner_model": result.get("planner_model", ""),
            "prompt": result.get("prompt", ""),
            "user_prompt": result.get("user_prompt", ""),
            "aspect_ratio": result.get("aspect_ratio", ""),
            "duration": result.get("duration", ""),
            "duration_sec": result.get("duration_sec", 8),
            "resolution": result.get("resolution", "720p"),
            "minimum_beats": result.get("minimum_beats", 0),
            "narration_chars": result.get("narration_chars", 0),
            "voice_selection": result.get("voice_selection", ""),
            "voice_provider": result.get("voice_provider", ""),
            "audio_duration_sec": result.get("audio_duration_sec", 0),
            "audio_complete": bool(result.get("audio_complete", not bool(result.get("generate_audio", False)))),
            "generate_audio": bool(result.get("generate_audio", False)),
            "brand_name": result.get("brand_name", ""),
            "provider": result.get("provider", "fal_veo31lite_ai_product_ads_text_video"),
            "pipeline_mode": "Rep_sdt2v_AI__Ads_video_cpu",
            "error_message": "",
        }

        _send_callback(callback_url, token, output)
        log(f"=== JOB COMPLETED: {job_id} ===")
        return output

    except Exception as e:
        err = str(e)
        tb = traceback.format_exc()
        log(f"ERROR: {err}")
        log(tb)
        output = {
            "ok": False,
            "job_id": job_id,
            "status": "failed",
            "step": "failed",
            "progress_pct": 100,
            "result_video_key": "",
            "result_video_url": "",
            "provider": "fal_veo31lite_ai_product_ads_text_video",
            "pipeline_mode": "Rep_sdt2v_AI__Ads_video_cpu",
            "error_message": err,
            "traceback": tb[-4000:],
        }
        _send_callback(callback_url, token, output)
        return output
    finally:
        # Preserve the original external job_id contract. Cleanup is scoped only when
        # the request supplied a real job_id, so malformed/id-less jobs cannot delete
        # temp files belonging to another concurrent invocation.
        if has_real_job_id:
            _cleanup_job_temp_files(job_id)
        else:
            log("WARNING: temp cleanup skipped because request had no job_id")
        log(f"JOB RELEASED | job={job_id} | total_elapsed={time.perf_counter() - job_started:.2f}s")


def _read_worker_concurrency() -> int:
    raw = os.getenv("RUNPOD_WORKER_CONCURRENCY", "1")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, 32))


WORKER_CONCURRENCY = _read_worker_concurrency()
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY, thread_name_prefix="ads-job")


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # The ads pipeline is blocking (requests + ffmpeg). Give it a dedicated executor whose
    # size is exactly the per-worker concurrency configured in RunPod ENV.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(JOB_EXECUTOR, _handle_job, event)


def _configured_concurrency(_: int) -> int:
    return WORKER_CONCURRENCY


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": _configured_concurrency,
})

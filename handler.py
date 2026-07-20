from __future__ import annotations

"""
RunPod Serverless handler — Rep_sdt2v_AI__Ads_video_cpu.
CPU orchestrator: GPT-4o-mini planner -> duration-specific Replicate Seedance source -> slow motion -> optional TTS mux -> R2 -> Cloudflare callback.
"""

import os
import traceback
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


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    job: Dict[str, Any] = event.get("input", {})
    job_id: str = str(job.get("job_id") or "unknown")
    callback_url = str(job.get("callback_url") or os.getenv("CALLBACK_URL") or "").strip()
    token = os.getenv("INTERNAL_CALLBACK_TOKEN", "")

    log(f"=== JOB START: {job_id} ===")
    log(f"Job type: {job.get('job_type')} | pipeline_mode: {job.get('pipeline_mode')}")

    try:
        result = generate_rep_sdt2v_ai_ads_video(job)
        local_path: str = result["local_path"]

        log(f"Pipeline done. Local path: {local_path}")

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
            "replicate_video_url": result.get("replicate_video_url", ""),
            "replicate_model": result.get("replicate_model", ""),
            "replicate_source_duration": result.get("replicate_source_duration", ""),
            "postprocess_extend": result.get("postprocess_extend", {}),
            "planner_model": result.get("planner_model", ""),
            "prompt": result.get("prompt", ""),
            "user_prompt": result.get("user_prompt", ""),
            "aspect_ratio": result.get("aspect_ratio", ""),
            "duration": result.get("duration", ""),
            "duration_sec": result.get("duration_sec", 8),
            "resolution": result.get("resolution", "480p"),
            "minimum_beats": result.get("minimum_beats", 0),
            "narration_chars": result.get("narration_chars", 0),
            "voice_selection": result.get("voice_selection", ""),
            "voice_provider": result.get("voice_provider", ""),
            "audio_duration_sec": result.get("audio_duration_sec", 0),
            "generate_audio": bool(result.get("generate_audio", False)),
            "provider": "replicate_seedance_ai_product_ads_text_video",
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
            "provider": "replicate_seedance_ai_product_ads_text_video",
            "pipeline_mode": "Rep_sdt2v_AI__Ads_video_cpu",
            "error_message": err,
            "traceback": tb[-4000:],
        }
        _send_callback(callback_url, token, output)
        return output


runpod.serverless.start({"handler": handler})

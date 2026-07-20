# SETUP GUIDE — Rep_sdt2v_AI__Ads_video_cpu

Endpoint này tạo **video quảng cáo sản phẩm cao cấp từ văn bản**. Luồng được giữ cùng cấu trúc xử lý với endpoint `Rep_sdt2v_AI_video_cpu`: AI Planner tối ưu mô tả, Replicate Seedance tạo video nguồn không audio, ffmpeg kéo chậm tới đúng thời lượng và TTS chỉ được tạo khi người dùng nhập lời thoại.

Điểm khác biệt riêng của endpoint này là prompt cuối gửi sang Replicate luôn được bổ sung chính xác một lần câu mặc định sau:

> Create a professional product advertising video.

Tổng prompt cuối vẫn không vượt quá **1500 ký tự**.

## Quy tắc thời lượng cố định

| Video đầu ra | Video nguồn Replicate | Resolution | Audio từ Replicate | Beats tối thiểu | Giới hạn lời thoại |
|---|---:|---|---|---:|---:|
| 8s | 4s | 480p | false | 5 | 120 ký tự |
| 15s | 4s | 480p | false | 7 | 225 ký tự |
| 30s | 5s | 480p | false | 9 | 450 ký tự |
| 60s | 8s | 480p | false | 14 | 900 ký tự |

Model được khóa trong pipeline là `bytedance/seedance-2.0-mini`.

Prompt cuối luôn chứa đúng một lần ba câu bắt buộc: yêu cầu quảng cáo sản phẩm chuyên nghiệp, chuyển động tự nhiên/không lỗi AI và chất lượng hình ảnh sắc nét 4K. Hàm chuẩn hóa sẽ loại bản lặp trước khi gắn lại ba câu nên không phát sinh trùng lặp.

## Files chính

- `handler.py`
- `rep_sdt2v_ai_ads_video_pipeline.py`
- `r2_utils.py`
- `test_request_rep_sdt2v_ai_ads_video_cpu.json`

## Biến môi trường RunPod endpoint

Bắt buộc:

```text
REPLICATE_API_TOKEN=...
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=...
R2_PUBLIC_BASE_URL=...
INTERNAL_CALLBACK_TOKEN=...
```

Planner và GPT voice:

```text
OPENAI_API_KEY=...
OPENAI_PLANNER_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=gpt-4o-mini-tts
```

Voice public VN hoặc Voice clone dùng OmniVoice:

```text
RUNPOD_OMNIVOICE_ENDPOINT_URL=https://api.runpod.ai/v2/<omnivoice-endpoint-id>/run
RUNPOD_OMNIVOICE_API_KEY=rpa_xxxxx
```

`RUNPOD_API_KEY` có thể được dùng làm fallback cho OmniVoice nếu không khai báo key riêng.

Tuỳ chọn:

```text
REPLICATE_POLL_TIMEOUT_SEC=900
REPLICATE_POLL_INTERVAL_SEC=5
REPLICATE_EXTRA_INPUT_JSON={"camera_fixed":false}
OMNIVOICE_POLL_TIMEOUT_SEC=900
OMNIVOICE_POLL_INTERVAL_SEC=4
OMNIVOICE_SPEED=1.0
OMNIVOICE_NUM_STEP=32
```

`REPLICATE_EXTRA_INPUT_JSON` không thể ghi đè `duration`, `resolution` hoặc `generate_audio`, vì ba trường này được khóa theo rule của pipeline.

## Biến môi trường Cloudflare Worker

```text
RUNPOD_REP_SDT2V_AI_ADS_VIDEO_CPU_API_URL=https://api.runpod.ai/v2/<endpoint-id>/run
RUNPOD_REP_SDT2V_AI_ADS_VIDEO_CPU_API_KEY=rpa_xxxxx
INTERNAL_CALLBACK_TOKEN=...
```

## Callback

```text
/api/internal/rep-sdt2v-ai-ads-video-callback
```

## Định danh request

```text
job_type=rep_sdt2v_ai_ads_video_text_to_video
provider=replicate_seedance_ai_product_ads_text_video
mode=rep_sdt2v_ai_ads_video_t2v
pipeline_mode=Rep_sdt2v_AI__Ads_video_cpu
```

## Input test

Xem file `test_request_rep_sdt2v_ai_ads_video_cpu.json`.

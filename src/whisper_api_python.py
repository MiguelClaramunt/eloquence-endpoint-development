import io
import os
import subprocess

import librosa
import uvicorn
import whisperx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile


# --- Device / compute configuration -----------------------------------------
# The endpoint used to hard-code CUDA. The HPC target may have no GPU, so we
# auto-detect a usable CUDA GPU and fall back to CPU when none is present.
# Every knob can be forced via environment variable so the same image runs
# unchanged on GPU and CPU nodes:
#
#   WHISPER_DEVICE        "cuda" | "cpu"            (default: auto-detect)
#   WHISPER_COMPUTE_TYPE  e.g. "float16","int8"     (default: float16 cuda / int8 cpu)
#   WHISPER_BATCH_SIZE    int                       (default: 16 cuda / 1 cpu)
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    """Return True if a usable CUDA GPU is present.

    Two signals, in order:
      1. ``nvidia-smi`` exits 0   -> a GPU + working driver exist.
      2. ``torch.cuda.is_available()`` -> torch itself can see a device.
    Either is enough to select CUDA. Everything is wrapped so a broken or
    absent CUDA stack can never crash the endpoint at import time (a CPU-only
    HPC node has neither nvidia-smi nor a CUDA device).
    """
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        if proc.returncode == 0:
            return True
    except Exception:
        pass
    try:
        import torch  # lazy import: don't hard-fail when torch is unavailable
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device() -> str:
    """WHISPER_DEVICE overrides auto-detection; else cuda-if-present, else cpu."""
    forced = os.environ.get("WHISPER_DEVICE", "").strip().lower()
    if forced:
        if forced == "cuda" and not _cuda_available():
            print(
                "[whisper] WARNING: WHISPER_DEVICE=cuda was forced but no usable "
                "CUDA GPU was detected - model load will likely fail.",
                flush=True,
            )
        return forced
    return "cuda" if _cuda_available() else "cpu"


def _resolve_compute_type(device: str) -> str:
    """WHISPER_COMPUTE_TYPE overrides; else float16 on CUDA, int8 on CPU."""
    forced = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip()
    if forced:
        return forced
    return "float16" if device == "cuda" else "int8"


def _resolve_batch_size(device: str) -> int:
    """WHISPER_BATCH_SIZE overrides; else 16 on CUDA, 1 on CPU (memory-safe)."""
    forced = os.environ.get("WHISPER_BATCH_SIZE", "").strip()
    if forced:
        try:
            return max(1, int(forced))
        except ValueError:
            print(
                f"[whisper] WARNING: invalid WHISPER_BATCH_SIZE={forced!r}; ignoring.",
                flush=True,
            )
    return 16 if device == "cuda" else 1


# Device and model settings (resolved once, at import time - single instance)
device = _resolve_device()
compute_type = _resolve_compute_type(device)
batch_size = _resolve_batch_size(device)

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

app = FastAPI()


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        audio_bytes = io.BytesIO(content)
        audio, sr = librosa.load(audio_bytes, sr=None, mono=False)

        result = model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=500, detail="HF_TOKEN not configured in environment"
            )
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )

        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments_out = []
        for segment in result["segments"]:
            segments_out.append(
                {
                    "speaker": segment.get("speaker", "unknown"),
                    "text": segment["text"],
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                }
            )

        return {"segments": segments_out}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"WhisperX failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "whisper_api_python:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=["/src"],
    )

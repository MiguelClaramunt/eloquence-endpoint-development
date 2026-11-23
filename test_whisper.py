"""
End-to-end smoke test for src/whisper_api_python.py.

Loads the whisperx model exactly like the FastAPI app does and transcribes a
short generated tone through the SAME call path used by the endpoint, then
checks the returned shape. Requires a valid HF token for the WhisperX
alignment model (HF_TOKEN env var).

Run:  python3 test_whisper.py
"""
import importlib.util
import os
import sys
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent


def _load_module():
    path = REPO_ROOT / "src" / "whisper_api_python.py"
    spec = importlib.util.spec_from_file_location("whisper_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set - cannot load the WhisperX alignment model.")
        sys.exit(2)
    os.environ["HF_TOKEN"] = token

    print("Loading model (this downloads + initializes large-v2)...")
    t0 = time.time()
    mod = _load_module()
    print(f"Model loaded in {time.time() - t0:.1f}s "
          f"(device={mod.device}, compute_type={mod.compute_type}, "
          f"batch_size={mod.batch_size})")

    # Synthetic audio: 3s of a 440 Hz sine wave, 16 kHz mono (same as endpoint).
    sr = 16000
    t = np.linspace(0.0, 3.0, int(sr * 3.0), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio, sr = librosa.load(io_bytesio(wave, sr), sr=None, mono=False)

    print("Transcribing...")
    t0 = time.time()
    result = mod.model.transcribe(audio, batch_size=mod.batch_size)
    text = "".join(seg["text"] for seg in result["segments"])
    print(f"Transcription done in {time.time() - t0:.1f}s")
    print(f"Segments: {len(result['segments'])}")
    print(f"Text: {text!r}")

    # The tone has no speech, so the expected text may be empty/garbage - we
    # only assert the pipeline ran and returned a well-formed dict.
    assert isinstance(result, dict) and "segments" in result
    assert isinstance(result["segments"], list)
    print("\nPASS: endpoint transcription path works end-to-end.")


def io_bytesio(wave, sr):
    import io
    from scipy.io import wavfile
    buf = io.BytesIO()
    wavfile.write(buf, sr, wave)
    buf.seek(0)
    return buf


if __name__ == "__main__":
    main()

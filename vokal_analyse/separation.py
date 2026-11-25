"""Vocal / music separation with python-audio-separator (RoFormer, Demucs).

The model is downloaded on first use into the models folder. Results are
cached per input file (content hash), so analysing the same song again is fast.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

# Best open vocal model in audio-separator's list, then a fast fallback.
DEFAULT_MODELS = [
    "vocals_mel_band_roformer.ckpt",            # Mel-Band RoFormer (Kimberley Jensen), vocals
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # BS-RoFormer, vocals/instrumental
    "htdemucs_ft.yaml",                          # Demucs v4 fine-tuned
]


def file_hash(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


class StemSeparator:
    def __init__(self, model_dir: str | os.PathLike, cache_dir: str | os.PathLike,
                 models: Optional[list[str]] = None, log: Callable[[str], None] = print):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.models = models or list(DEFAULT_MODELS)
        self.log = log
        self._sep = None
        self._model = None

    def _load(self):
        if self._sep is not None:
            return
        import logging
        from audio_separator.separator import Separator  # heavy import, only when needed
        self.model_dir.mkdir(parents=True, exist_ok=True)
        work = self.cache_dir / "_work"
        work.mkdir(parents=True, exist_ok=True)
        last_err = None
        for name in self.models:
            try:
                self.log(f"Loading separation model {name} (downloads on first use)...")
                sep = Separator(model_file_dir=str(self.model_dir), output_dir=str(work),
                                output_format="WAV", log_level=logging.WARNING)
                sep.load_model(model_filename=name)
                self._sep, self._model = sep, name
                return
            except Exception as e:  # try the next model
                last_err = e
                self.log(f"Could not load {name}: {e}")
        raise RuntimeError(f"No separation model could be loaded: {last_err}")

    def separate(self, path: str | os.PathLike) -> tuple[Path, Path]:
        """Return (vocal wav, instrumental wav)."""
        key = file_hash(path)
        out = self.cache_dir / key
        voc, ins = out / "vocals.wav", out / "music.wav"
        if voc.exists() and ins.exists():
            self.log("Using cached separation.")
            return voc, ins
        self._load()
        self.log(f"Separating vocal from music with {self._model}. This takes a while on CPU, under a minute on a GPU...")
        files = self._sep.separate(str(path))
        work = Path(self._sep.output_dir)
        files = [Path(f) if Path(f).is_absolute() else work / f for f in files]
        vocal = [f for f in files if "(vocals)" in f.name.lower()]
        other = [f for f in files if f not in vocal]
        if not vocal:
            raise RuntimeError(f"The model gave no vocal stem: {[f.name for f in files]}")
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vocal[0]), voc)
        if len(other) == 1:
            shutil.move(str(other[0]), ins)
        else:
            # several non-vocal stems (Demucs): sum them into one music stem
            import numpy as np
            import soundfile as sf
            acc, sr = None, None
            for f in other:
                x, sr = sf.read(f, always_2d=True)
                acc = x if acc is None else acc[:len(x)] + x[:len(acc)]
                f.unlink()
            sf.write(ins, acc, sr)
        return voc, ins

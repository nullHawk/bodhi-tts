import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path


class MelProcessor:
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_mels=80, f_min=0, f_max=12000):
        self.sr = sr
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def compute(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute log-mel spectrogram from waveform tensor.

        Args:
            waveform: [C, T] or [T] audio tensor
            sample_rate: source sample rate

        Returns:
            mel: [n_mels, T_mel]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        # Resample
        if sample_rate != self.sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sr)
        mel = self.mel_transform(waveform)  # [1, n_mels, T]
        mel = torch.clamp(mel, min=1e-5).log()
        return mel.squeeze(0)  # [n_mels, T]

    def compute_and_cache(self, audio_path: str, cache_dir: str, sample_id: str) -> tuple:
        """Compute mel and cache as .pt file. Returns (mel_tensor, cache_path)."""
        cache_path = Path(cache_dir) / f"{sample_id}.pt"
        if cache_path.exists():
            mel = torch.load(cache_path, weights_only=True)
            return mel, str(cache_path)

        audio_np, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(audio_np).float()
        mel = self.compute(waveform, sr)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, cache_path)
        return mel, str(cache_path)

    def compute_from_array_and_cache(self, audio_array, sample_rate: int, cache_dir: str, sample_id: str) -> tuple:
        """Compute mel from numpy array and cache. Returns (mel_tensor, cache_path)."""
        cache_path = Path(cache_dir) / f"{sample_id}.pt"
        if cache_path.exists():
            mel = torch.load(cache_path, weights_only=True)
            return mel, str(cache_path)

        waveform = torch.from_numpy(audio_array).float()
        mel = self.compute(waveform, sample_rate)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, cache_path)
        return mel, str(cache_path)

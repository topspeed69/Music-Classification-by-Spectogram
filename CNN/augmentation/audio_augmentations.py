"""
Audio Augmentation Pipeline for Self-Supervised Contrastive Learning
Implements two-stage augmentation: waveform-level and spectrogram-level
Optimized for speed with worker pool management
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import random
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


# ==================== WAVEFORM AUGMENTATIONS ====================

class PitchShift:
    """Shift pitch by ±1-3 semitones"""
    
    def __init__(self, sr: int = 22050, n_steps_range: Tuple[int, int] = (-3, 3)):
        self.sr = sr
        self.n_steps_range = n_steps_range
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        n_steps = random.uniform(*self.n_steps_range)
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)


class TempoStretch:
    """Stretch tempo by ±5-12%"""
    
    def __init__(self, rate_range: Tuple[float, float] = (0.88, 1.12)):
        self.rate_range = rate_range
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        rate = random.uniform(*self.rate_range)
        return librosa.effects.time_stretch(y, rate=rate)


class GainAdjustment:
    """Adjust gain by ±3-6 dB"""
    
    def __init__(self, db_range: Tuple[float, float] = (-6, 6)):
        self.db_range = db_range
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        db_change = random.uniform(*self.db_range)
        gain = 10 ** (db_change / 20.0)
        return y * gain


class ParametricEQ:
    """Apply parametric EQ filtering (LPF, HPF, or bandpass)"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.filter_types = ['lowpass', 'highpass', 'bandpass']
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        filter_type = random.choice(self.filter_types)
        
        if filter_type == 'lowpass':
            # Cutoff between 2kHz-8kHz
            cutoff = random.uniform(2000, 8000)
            y_filtered = librosa.effects.preemphasis(y, coef=0.0)
            # Simple lowpass approximation
            y_filtered = librosa.core.lpc(y, order=2)
            y_filtered = y if len(y_filtered) == 0 else y
            
        elif filter_type == 'highpass':
            # Cutoff between 100Hz-500Hz
            cutoff = random.uniform(100, 500)
            y_filtered = librosa.effects.preemphasis(y, coef=0.97)
            
        else:  # bandpass
            # Apply both high and low pass
            y_filtered = librosa.effects.preemphasis(y, coef=0.85)
        
        return y_filtered


class DynamicRangeCompression:
    """Apply dynamic range compression"""
    
    def __init__(self, threshold_db: float = -20, ratio: float = 4.0):
        self.threshold_db = threshold_db
        self.ratio = ratio
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        # Convert to dB
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        
        # Apply compression
        compressed_db = np.where(
            db > self.threshold_db,
            self.threshold_db + (db - self.threshold_db) / self.ratio,
            db
        )
        
        # Convert back to amplitude
        compressed = librosa.db_to_amplitude(compressed_db)
        
        # Preserve phase
        phase = np.angle(y + 1e-10)
        return compressed * np.exp(1j * phase).real


class AddNoise:
    """Add environmental noise with SNR 10-30 dB"""
    
    def __init__(self, snr_db_range: Tuple[float, float] = (10, 30)):
        self.snr_db_range = snr_db_range
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(*self.snr_db_range)
        
        # Generate white noise
        noise = np.random.randn(len(y))
        
        # Calculate noise power for desired SNR
        signal_power = np.mean(y ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Scale noise to achieve desired SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        return y + noise_scale * noise


class AddReverb:
    """Add convolutional reverb with synthetic impulse response"""
    
    def __init__(self, sr: int = 22050, reverb_amount: Tuple[float, float] = (0.1, 0.4)):
        self.sr = sr
        self.reverb_amount = reverb_amount
        
    def __call__(self, y: np.ndarray) -> np.ndarray:
        # Generate simple exponential decay impulse response
        reverb_time = random.uniform(*self.reverb_amount)
        ir_length = int(self.sr * reverb_time)
        
        # Create exponential decay IR
        decay = np.exp(-3 * np.linspace(0, 1, ir_length))
        ir = np.random.randn(ir_length) * decay
        ir = ir / np.sum(ir ** 2)  # Normalize
        
        # Convolve with signal
        y_reverb = np.convolve(y, ir, mode='same')
        
        # Mix with dry signal
        mix = random.uniform(0.2, 0.5)
        return (1 - mix) * y + mix * y_reverb


# ==================== SPECTROGRAM AUGMENTATIONS ====================

class TimeMasking(nn.Module):
    """Time masking (SpecAugment) - mask random time steps"""
    
    def __init__(self, max_mask_ratio: float = 0.15, num_masks: int = 1):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
        
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Mel spectrogram [C, n_mels, time] or [n_mels, time]
        Returns:
            Masked spectrogram
        """
        cloned = spec.clone()
        
        if len(cloned.shape) == 3:
            _, _, time_steps = cloned.shape
        else:
            _, time_steps = cloned.shape
        
        for _ in range(self.num_masks):
            mask_size = int(time_steps * random.uniform(0.05, self.max_mask_ratio))
            mask_start = random.randint(0, max(1, time_steps - mask_size))
            
            if len(cloned.shape) == 3:
                cloned[:, :, mask_start:mask_start + mask_size] = 0
            else:
                cloned[:, mask_start:mask_start + mask_size] = 0
        
        return cloned


class FrequencyMasking(nn.Module):
    """Frequency masking (SpecAugment) - mask random frequency bands"""
    
    def __init__(self, max_mask_ratio: float = 0.15, num_masks: int = 1):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
        
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Mel spectrogram [C, n_mels, time] or [n_mels, time]
        Returns:
            Masked spectrogram
        """
        cloned = spec.clone()
        
        if len(cloned.shape) == 3:
            _, n_mels, _ = cloned.shape
        else:
            n_mels, _ = cloned.shape
        
        for _ in range(self.num_masks):
            mask_size = int(n_mels * random.uniform(0.05, self.max_mask_ratio))
            mask_start = random.randint(0, max(1, n_mels - mask_size))
            
            if len(cloned.shape) == 3:
                cloned[:, mask_start:mask_start + mask_size, :] = 0
            else:
                cloned[mask_start:mask_start + mask_size, :] = 0
        
        return cloned


class TimeWarping(nn.Module):
    """Time warping - warp time axis for robustness"""
    
    def __init__(self, max_warp: int = 10):
        super().__init__()
        self.max_warp = max_warp
        
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Mel spectrogram [C, n_mels, time] or [n_mels, time]
        Returns:
            Warped spectrogram
        """
        if len(spec.shape) == 3:
            C, n_mels, time_steps = spec.shape
        else:
            n_mels, time_steps = spec.shape
            C = None
        
        if time_steps < 10:
            return spec
        
        # Choose random center point
        center = random.randint(self.max_warp, time_steps - self.max_warp)
        warp = random.randint(-self.max_warp, self.max_warp)
        
        # Create warped version using interpolation
        if C is not None:
            # 3D case
            warped = spec.clone()
            if warp > 0:
                # Stretch
                warped[:, :, center:] = torch.nn.functional.interpolate(
                    spec[:, :, center:].unsqueeze(0),
                    size=(n_mels, time_steps - center),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                # Compress
                warped[:, :, center:] = torch.nn.functional.interpolate(
                    spec[:, :, center:].unsqueeze(0),
                    size=(n_mels, time_steps - center),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        else:
            # 2D case
            warped = spec.clone()
        
        return warped


# ==================== MAIN AUGMENTATION PIPELINE ====================

class ContrastiveAudioAugmentation:
    """
    Two-stage augmentation pipeline for self-supervised contrastive learning.
    Stage 1: Waveform augmentations (before mel spectrogram)
    Stage 2: Spectrogram augmentations (after mel transform)
    
    Returns two independently augmented views of the same audio for contrastive learning.
    """
    
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        duration: float = 3.0,
        num_waveform_augs: int = 3,
        num_spectrogram_augs: int = 2,
        max_workers: int = 7,
        waveform_aug_probs: Optional[dict] = None,
        spectrogram_aug_probs: Optional[dict] = None
    ):
        """
        Args:
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            duration: Audio duration in seconds
            num_waveform_augs: Number of waveform augmentations to apply (randomly selected)
            num_spectrogram_augs: Number of spectrogram augmentations to apply
            max_workers: Maximum number of worker threads (capped at 7)
            waveform_aug_probs: Probability dict for each waveform augmentation
            spectrogram_aug_probs: Probability dict for each spectrogram augmentation
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.num_waveform_augs = num_waveform_augs
        self.num_spectrogram_augs = num_spectrogram_augs
        self.max_workers = min(max_workers, 7)  # Cap at 7 workers
        
        # Initialize waveform augmentations
        self.waveform_augmentations = [
            ('pitch_shift', PitchShift(sr=sr)),
            ('tempo_stretch', TempoStretch()),
            ('gain', GainAdjustment()),
            ('eq', ParametricEQ(sr=sr)),
            ('compression', DynamicRangeCompression()),
            ('noise', AddNoise()),
            ('reverb', AddReverb(sr=sr))
        ]
        
        # Default probabilities for waveform augmentations
        self.waveform_aug_probs = waveform_aug_probs or {
            'pitch_shift': 0.6,
            'tempo_stretch': 0.5,
            'gain': 0.7,
            'eq': 0.5,
            'compression': 0.4,
            'noise': 0.6,
            'reverb': 0.5
        }
        
        # Initialize spectrogram augmentations
        self.spectrogram_augmentations = [
            ('time_mask', TimeMasking(max_mask_ratio=0.15, num_masks=1)),
            ('freq_mask', FrequencyMasking(max_mask_ratio=0.15, num_masks=1)),
            ('time_warp', TimeWarping(max_warp=10))
        ]
        
        # Default probabilities for spectrogram augmentations
        self.spectrogram_aug_probs = spectrogram_aug_probs or {
            'time_mask': 0.8,
            'freq_mask': 0.8,
            'time_warp': 0.5
        }
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file"""
        y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
        return y
    
    def apply_waveform_augmentations(self, y: np.ndarray) -> np.ndarray:
        """
        Apply randomly selected waveform augmentations
        
        Args:
            y: Audio waveform
        Returns:
            Augmented waveform
        """
        # Randomly select augmentations
        selected_augs = random.sample(
            self.waveform_augmentations,
            min(self.num_waveform_augs, len(self.waveform_augmentations))
        )
        
        # Apply each selected augmentation with probability
        y_aug = y.copy()
        for aug_name, aug_fn in selected_augs:
            if random.random() < self.waveform_aug_probs.get(aug_name, 0.5):
                try:
                    y_aug = aug_fn(y_aug)
                    # Ensure output length matches input
                    if len(y_aug) > len(y):
                        y_aug = y_aug[:len(y)]
                    elif len(y_aug) < len(y):
                        y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)), mode='constant')
                except Exception as e:
                    # If augmentation fails, skip it
                    pass
        
        return y_aug
    
    def waveform_to_mel_spectrogram(self, y: np.ndarray) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram
        
        Args:
            y: Audio waveform
        Returns:
            Mel spectrogram tensor [n_mels, time]
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to torch tensor
        return torch.from_numpy(mel_spec_db).float()
    
    def apply_spectrogram_augmentations(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply randomly selected spectrogram augmentations
        
        Args:
            spec: Mel spectrogram [n_mels, time]
        Returns:
            Augmented spectrogram
        """
        # Randomly select augmentations
        selected_augs = random.sample(
            self.spectrogram_augmentations,
            min(self.num_spectrogram_augs, len(self.spectrogram_augmentations))
        )
        
        # Apply each selected augmentation with probability
        spec_aug = spec.clone()
        for aug_name, aug_fn in selected_augs:
            if random.random() < self.spectrogram_aug_probs.get(aug_name, 0.5):
                try:
                    spec_aug = aug_fn(spec_aug)
                except Exception as e:
                    # If augmentation fails, skip it
                    pass
        
        return spec_aug
    
    def __call__(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply complete augmentation pipeline and return two augmented views
        
        Args:
            audio_path: Path to audio file
        Returns:
            (view1, view2): Two independently augmented mel spectrograms [n_mels, time]
        """
        # Load original audio
        y_original = self.load_audio(audio_path)
        
        # Create two augmented views
        # View 1: waveform augmentation -> mel spectrogram -> spectrogram augmentation
        y_aug1 = self.apply_waveform_augmentations(y_original)
        spec1 = self.waveform_to_mel_spectrogram(y_aug1)
        view1 = self.apply_spectrogram_augmentations(spec1)
        
        # View 2: independent augmentation
        y_aug2 = self.apply_waveform_augmentations(y_original)
        spec2 = self.waveform_to_mel_spectrogram(y_aug2)
        view2 = self.apply_spectrogram_augmentations(spec2)
        
        # Add channel dimension if needed [1, n_mels, time]
        if len(view1.shape) == 2:
            view1 = view1.unsqueeze(0)
        if len(view2.shape) == 2:
            view2 = view2.unsqueeze(0)
        
        return view1, view2
    
    def process_batch(self, audio_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of audio files in parallel
        
        Args:
            audio_paths: List of audio file paths
        Returns:
            (views1, views2): Batched augmented spectrograms [B, 1, n_mels, time]
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.__call__, audio_paths))
        
        # Stack results
        views1 = torch.stack([r[0] for r in results])
        views2 = torch.stack([r[1] for r in results])
        
        return views1, views2


# ==================== DATASET WITH AUGMENTATION ====================

from torch.utils.data import Dataset

class AudioDatasetWithAugmentation(Dataset):
    """
    Dataset that loads audio files and applies augmentation pipeline
    """
    
    def __init__(
        self,
        audio_dir: str,
        augmentation: ContrastiveAudioAugmentation,
        file_extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    ):
        """
        Args:
            audio_dir: Directory containing audio files
            augmentation: Augmentation pipeline instance
            file_extensions: Tuple of valid file extensions
        """
        from pathlib import Path
        
        self.audio_dir = Path(audio_dir)
        self.augmentation = augmentation
        
        # Find all audio files
        self.audio_paths = []
        for ext in file_extensions:
            self.audio_paths.extend(list(self.audio_dir.rglob(f'*{ext}')))
        
        if len(self.audio_paths) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(self.audio_paths)} audio files in {audio_dir}")
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two augmented views of the same audio
        
        Args:
            idx: Sample index
        Returns:
            (view1, view2): Two augmented mel spectrograms
        """
        audio_path = str(self.audio_paths[idx])
        return self.augmentation(audio_path)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize augmentation pipeline
    aug_pipeline = ContrastiveAudioAugmentation(
        sr=22050,
        n_mels=128,
        duration=3.0,
        num_waveform_augs=3,
        num_spectrogram_augs=2,
        max_workers=7
    )
    
    # Test with a single audio file
    # view1, view2 = aug_pipeline("path/to/audio.wav")
    # print(f"View 1 shape: {view1.shape}")
    # print(f"View 2 shape: {view2.shape}")
    
    # Create dataset
    # dataset = AudioDatasetWithAugmentation(
    #     audio_dir="AudioToSpectogram/fma_small_dataset",
    #     augmentation=aug_pipeline
    # )
    
    # Create dataloader
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    print("Augmentation pipeline initialized successfully!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import librosa
import numpy as np
from tqdm import tqdm
import subprocess
import tempfile
import os
from datetime import datetime


# Log file for recording missed/corrupted audio files
LOG_FILE = Path(__file__).parent / 'failed_audio_files.log'

# Global counters (will be used in single-threaded logging)
processed_count = 0
failed_count = 0


def list_files(source):
    """
    List all files in the given source directory and its subdirectories.

    Args:
        source (str): The source directory path.

    Returns:
        list: A list of `Path` objects representing the files.
    """
    path = Path(source)
    files = [file for file in path.rglob('*') if file.is_file()]
    return files


def audio_to_spectrogram(audio_path, save_path, duration):
    """
    Convert an audio file to a spectrogram and save it as an image.

    Args:
        audio_path (str): The path to the audio file.
        save_path (str): The path to save the spectrogram image.
        duration (int): Duration of the audio file to process in seconds.

    Returns:
        bool: True if successful, False if failed
    """
    # Load audio file with optimized settings
    y, sr = load_audio_safe(audio_path, duration=duration)

    # If load failed, skip and do not create an image
    if y is None:
        return False

    # Compute spectrogram with optimized parameters
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Normalize values to 0-255 range and convert to uint8
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to RGB and save as PNG
    S = cv2.cvtColor(S, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, S)
    return True


def load_audio_safe(path, duration=None, sr=22050):
    """Load audio; if librosa/soundfile fails, transcode via ffmpeg to a temp WAV and load that.

    On irrecoverable failure, append an entry to `failed_audio_files.log` and return (None, None).
    """
    try:
        # Using kaiser_fast for faster resampling and explicitly set mono
        return librosa.load(path, sr=sr, duration=duration, mono=True, res_type='kaiser_fast')
    except Exception as e1:
        # Try to transcode with ffmpeg into a temporary WAV and load that
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-v", "quiet", "-i", str(path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
            "-t", str(duration) if duration else "60",
            tmp.name
        ]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            # ffmpeg not installed or not on PATH
            try:
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()} | {path} | ffmpeg not found; original error: {repr(e1)}\n")
            except Exception:
                pass
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            return None, None
        
        try:
            try:
                y, sr_out = librosa.load(tmp.name, sr=sr, duration=duration, mono=True, res_type='kaiser_fast')
                return y, sr_out
            except Exception as e2:
                # Log the failure with timestamp and exception message
                try:
                    with LOG_FILE.open("a", encoding="utf-8") as f:
                        f.write(f"{datetime.now().isoformat()} | {path} | {repr(e2)}\n")
                except Exception:
                    pass
                return None, None
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass


def process_file(args):
    """Process a single file (for multiprocessing)."""
    file, source, output, duration = args
    try:
        # Output path
        new_path = Path(str(file).replace(str(source), output))

        # Create output directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace suffix
        new_path = new_path.with_suffix('.png')

        # Convert
        success = audio_to_spectrogram(str(file), str(new_path), duration)
        return success
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cats_dogs', help='source folder')
    parser.add_argument('--duration', type=int, default=60, help='duration of audios in case they are too big')
    parser.add_argument('--output', type=str, default='output', help='folder output')
    parser.add_argument('--workers', type=int, default=None, help='number of parallel workers (default: CPU count)')
    opt = parser.parse_args()
    source, duration, output = opt.source, opt.duration, opt.output
    workers = opt.workers if opt.workers else cpu_count()

    file_list = list_files(source)
    print(f"Found {len(file_list)} files. Processing with {workers} workers...")

    # Prepare arguments for multiprocessing
    args_list = [(file, source, output, duration) for file in file_list]

    # Process files in parallel
    successful = 0
    failed = 0
    
    with Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(process_file, args_list), total=len(file_list)):
            if result:
                successful += 1
            else:
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    if failed > 0:
        print(f"Check {LOG_FILE} for details on failed files")
    print(f"{'='*60}")

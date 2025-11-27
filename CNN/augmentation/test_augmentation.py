"""
Test script to verify audio augmentation pipeline with fma_small_dataset
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_dataset_structure():
    """Test that fma_small_dataset exists and has audio files"""
    print("="*60)
    print("TEST 1: Dataset Structure")
    print("="*60)
    
    dataset_path = Path("../../AudioToSpectogram/fma_small_dataset")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"✓ Dataset directory exists: {dataset_path}")
    
    # Find audio files
    audio_files = list(dataset_path.glob("*.mp3"))
    audio_files.extend(list(dataset_path.glob("*.wav")))
    audio_files.extend(list(dataset_path.glob("*.flac")))
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return False
    
    # Show first few files
    print("\nFirst 5 audio files:")
    for i, f in enumerate(audio_files[:5]):
        print(f"  {i+1}. {f.name}")
    
    return True


def test_augmentation_import():
    """Test importing augmentation modules"""
    print("\n" + "="*60)
    print("TEST 2: Import Augmentation Modules")
    print("="*60)
    
    try:
        from augmentation.audio_augmentations import (
            ContrastiveAudioAugmentation,
            AudioDatasetWithAugmentation
        )
        print("✓ Successfully imported ContrastiveAudioAugmentation")
        print("✓ Successfully imported AudioDatasetWithAugmentation")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_augmentation_pipeline():
    """Test creating augmentation pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Initialize Augmentation Pipeline")
    print("="*60)
    
    try:
        from augmentation.audio_augmentations import ContrastiveAudioAugmentation
        
        aug = ContrastiveAudioAugmentation(
            sr=22050,
            n_mels=128,
            duration=3.0,
            num_waveform_augs=3,
            num_spectrogram_augs=2,
            max_workers=7
        )
        print("✓ Augmentation pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_single_audio_processing():
    """Test processing a single audio file"""
    print("\n" + "="*60)
    print("TEST 4: Process Single Audio File")
    print("="*60)
    
    try:
        from augmentation.audio_augmentations import ContrastiveAudioAugmentation
        
        # Find first audio file
        dataset_path = Path("../../AudioToSpectogram/fma_small_dataset")
        audio_files = list(dataset_path.glob("*.mp3"))
        
        if len(audio_files) == 0:
            print("❌ No audio files to test with")
            return False
        
        test_file = str(audio_files[0])
        print(f"Testing with: {audio_files[0].name}")
        
        # Create augmentation
        aug = ContrastiveAudioAugmentation(
            sr=22050,
            n_mels=128,
            duration=3.0,
            num_waveform_augs=2,  # Reduce for faster testing
            num_spectrogram_augs=1,
            max_workers=4
        )
        
        # Process file
        view1, view2 = aug(test_file)
        
        print(f"✓ Successfully generated two views")
        print(f"  View 1 shape: {view1.shape}")
        print(f"  View 2 shape: {view2.shape}")
        
        # Verify shapes
        assert len(view1.shape) == 3, "View should be 3D"
        assert view1.shape[0] == 1, "Should have 1 channel"
        assert view1.shape[1] == 128, "Should have 128 mel bands"
        
        print("✓ Output shapes are correct")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test creating PyTorch dataset"""
    print("\n" + "="*60)
    print("TEST 5: Create PyTorch Dataset")
    print("="*60)
    
    try:
        from augmentation.audio_augmentations import (
            ContrastiveAudioAugmentation,
            AudioDatasetWithAugmentation
        )
        
        # Create augmentation
        aug = ContrastiveAudioAugmentation(
            sr=22050,
            n_mels=128,
            duration=3.0,
            num_waveform_augs=2,
            num_spectrogram_augs=1,
            max_workers=4
        )
        
        # Create dataset
        dataset = AudioDatasetWithAugmentation(
            audio_dir="../../AudioToSpectogram/fma_small_dataset",
            augmentation=aug
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test getting one sample
        print("Getting first sample...")
        view1, view2 = dataset[0]
        
        print(f"✓ Successfully retrieved sample")
        print(f"  View 1 shape: {view1.shape}")
        print(f"  View 2 shape: {view2.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_val_split():
    """Test automatic train/val split"""
    print("\n" + "="*60)
    print("TEST 6: Train/Val Split")
    print("="*60)
    
    try:
        from augmentation.audio_augmentations import (
            ContrastiveAudioAugmentation,
            AudioDatasetWithAugmentation
        )
        from sklearn.model_selection import train_test_split
        import torch
        from torch.utils.data import Subset
        
        # Create augmentation
        aug = ContrastiveAudioAugmentation(
            sr=22050,
            n_mels=128,
            duration=3.0,
            num_waveform_augs=2,
            num_spectrogram_augs=1,
            max_workers=4
        )
        
        # Create full dataset
        full_dataset = AudioDatasetWithAugmentation(
            audio_dir="../../AudioToSpectogram/fma_small_dataset",
            augmentation=aug
        )
        
        # Split
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)
        
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        print(f"✓ Split dataset into:")
        print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Split failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AUDIO AUGMENTATION PIPELINE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Dataset Structure", test_dataset_structure()))
    results.append(("Import Modules", test_augmentation_import()))
    results.append(("Initialize Pipeline", test_augmentation_pipeline()))
    results.append(("Process Single Audio", test_single_audio_processing()))
    results.append(("Create Dataset", test_dataset_creation()))
    results.append(("Train/Val Split", test_train_val_split()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! The augmentation pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Run the demo: python demo_augmentation.py")
        print("  2. Start training: python train_with_audio_aug.py --help")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

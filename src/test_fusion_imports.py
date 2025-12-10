"""
Quick test to verify all fusion imports and classes work correctly.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from fusion_model import FusionModel
        print("✓ FusionModel imported")
    except Exception as e:
        print(f"✗ FusionModel import failed: {e}")
        return False
    
    try:
        from loaders import get_dataloaders, DynamicDataset
        print("✓ Loaders imported")
    except Exception as e:
        print(f"✗ Loaders import failed: {e}")
        return False
    
    try:
        from spatial_model import SpatialBranch
        print("✓ SpatialBranch imported")
    except Exception as e:
        print(f"✗ SpatialBranch import failed: {e}")
        return False
    
    try:
        from freq_model import FrequencyBranch
        print("✓ FrequencyBranch imported")
    except Exception as e:
        print(f"✗ FrequencyBranch import failed: {e}")
        return False
    
    return True


def test_models():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")
    import torch
    from spatial_model import SpatialBranch
    from freq_model import FrequencyBranch
    from fusion_model import FusionModel
    
    try:
        spatial = SpatialBranch(num_classes=2)
        print("✓ SpatialBranch instantiated")
    except Exception as e:
        print(f"✗ SpatialBranch instantiation failed: {e}")
        return False
    
    try:
        freq = FrequencyBranch(num_classes=2)
        print("✓ FrequencyBranch instantiated")
    except Exception as e:
        print(f"✗ FrequencyBranch instantiation failed: {e}")
        return False
    
    try:
        fusion = FusionModel(num_classes=2)
        print("✓ FusionModel instantiated")
    except Exception as e:
        print(f"✗ FusionModel instantiation failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    import torch
    from fusion_model import FusionModel
    
    try:
        model = FusionModel(num_classes=2)
        rgb = torch.randn(2, 3, 224, 224)
        fft = torch.randn(2, 1, 224, 224)
        
        output = model(rgb, fft)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        if output.shape == (2, 2):
            print("✓ Output shape is correct")
            return True
        else:
            print(f"✗ Output shape incorrect: expected (2, 2), got {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_fusion_mode():
    """Test that dataloader supports fusion mode."""
    print("\nTesting dataloader fusion mode...")
    from loaders import DynamicDataset
    
    try:
        # Try creating a dataset with fusion mode (even if no real data)
        dataset = DynamicDataset(roots=[], mode="fusion")
        print("✓ DynamicDataset supports 'fusion' mode")
        return True
    except Exception as e:
        print(f"✗ DynamicDataset fusion mode failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Fusion Training Setup Validation")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_models()
    all_passed &= test_forward_pass()
    all_passed &= test_dataloader_fusion_mode()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validation tests passed!")
        print("The fusion training setup is ready to go.")
    else:
        print("✗ Some validation tests failed.")
        print("Please review the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

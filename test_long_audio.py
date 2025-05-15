"""
Test script for verifying the long audio functionality.
This script tests the imports and initializes the key components
without actually generating audio.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Test if all necessary modules can be imported."""
    print("Testing imports...")
    
    # Import the long audio module
    try:
        from app.utils.long_audio import (
            generate_long_audio,
            generate_and_save_long_audio,
            generate_long_audio_api,
            split_text_into_chunks
        )
        print("✅ Long audio module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import long audio module: {e}")
        return False
    
    # Import the configuration
    try:
        from config import config_manager
        print("✅ Configuration manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import configuration manager: {e}")
        return False
    
    # Import the models
    try:
        from models import LongAudioRequest
        print("✅ LongAudioRequest model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LongAudioRequest model: {e}")
        return False
    
    return True

def test_functionality():
    """Test basic functionality without actually generating audio."""
    print("\nTesting functionality...")
    
    # Test text chunking
    try:
        from app.utils.long_audio import split_text_into_chunks
        test_text = "[S1] This is a test sentence. Here is another sentence. [S2] And here's a response."
        chunks = split_text_into_chunks(test_text, max_chars=50)
        print(f"✅ Text chunking works: Split into {len(chunks)} chunks")
        print(f"   Chunks: {chunks}")
    except Exception as e:
        print(f"❌ Text chunking failed: {e}")
        return False
    
    # Check for required directories
    try:
        from config import get_reference_audio_path, get_output_path
        ref_path = get_reference_audio_path()
        out_path = get_output_path()
        
        print(f"✅ Reference audio path: {ref_path}")
        print(f"✅ Output path: {out_path}")
        
        os.makedirs(ref_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)
        
        print("✅ Directories exist or were created successfully")
    except Exception as e:
        print(f"❌ Directory check failed: {e}")
        return False
    
    return True

def test_api_model():
    """Test the API model validation."""
    print("\nTesting API model validation...")
    
    try:
        from models import LongAudioRequest
        from pydantic import ValidationError
        
        # Test valid request
        valid_data = {
            "text": "Test text",
            "voice_mode": "single_s1",
            "output_format": "opus",
            "chunk_size": 300
        }
        
        request = LongAudioRequest(**valid_data)
        print("✅ Valid request passed validation")
        
        # Test invalid request (should raise ValidationError)
        invalid_data = {
            "text": "Test text",
            "voice_mode": "invalid_mode",  # Invalid value
            "chunk_size": 50  # Below minimum
        }
        
        try:
            request = LongAudioRequest(**invalid_data)
            print("❌ Invalid request passed validation (should have failed)")
            return False
        except ValidationError:
            print("✅ Invalid request correctly rejected")
        
    except Exception as e:
        print(f"❌ API model validation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running tests for long audio functionality...\n")
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import tests failed. Stopping further tests.")
        return 1
    
    func_ok = test_functionality()
    if not func_ok:
        print("\n❌ Functionality tests failed.")
        return 1
    
    api_ok = test_api_model()
    if not api_ok:
        print("\n❌ API model tests failed.")
        return 1
    
    print("\n✅ All tests passed successfully!")
    print("\nThe long audio functionality appears to be correctly set up.")
    print("You can now use the long audio features through the API or command line.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

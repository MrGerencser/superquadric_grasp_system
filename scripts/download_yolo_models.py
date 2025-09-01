#!/usr/bin/env python3
"""
Script to download YOLO models from Hugging Face Hub
"""

from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    """Download YOLO models"""
    
    # Set target directory relative to script location
    script_dir = Path(__file__).parent
    target_dir = script_dir.parent / "superquadric_grasp_system" / "models" / "yolo"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    repo_id = "Gerencser/yolo-detection-models"
    models = [
        "all_objects.pt",
        "cubes.pt"
    ]
    
    print(f"Downloading YOLO models to: {target_dir}")
    
    for filename in models:
        try:
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                repo_type="model"
            )
            print(f"✅ Downloaded {filename}")
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    print("🎉 All YOLO models downloaded!")


if __name__ == "__main__":
    main()
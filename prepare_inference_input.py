#!/usr/bin/env python3
"""Convert regular RGB images to nnU-Net input format."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_to_nnunet_format(input_dir: Path, output_dir: Path) -> None:
    """
    Convert RGB images to nnU-Net format (separate channels).
    
    Args:
        input_dir: Directory containing RGB images
        output_dir: Directory to save converted images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    print(f"🔄 Converting to nnU-Net format...")
    
    for idx, img_path in enumerate(sorted(image_files)):
        # Read RGB image
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        # Case identifier
        case_id = f"case_{idx:04d}"
        
        # Split into R, G, B channels and save separately
        for channel_idx in range(3):
            channel_data = img_array[:, :, channel_idx]
            channel_img = Image.fromarray(channel_data)
            
            # nnU-Net naming: {case_id}_{channel}.png
            output_path = output_dir / f"{case_id}_{channel_idx:04d}.png"
            channel_img.save(output_path)
        
        print(f"  ✅ {img_path.name} → {case_id}_0000.png (R), _0001.png (G), _0002.png (B)")
    
    print(f"\n✅ Conversion complete!")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Generated {len(image_files) * 3} channel files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RGB images to nnU-Net input format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing RGB images"
    )
    parser.add_argument(
        "output_dir", 
        type=Path,
        help="Directory to save nnU-Net format images"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    convert_to_nnunet_format(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

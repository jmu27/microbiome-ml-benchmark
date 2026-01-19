#!/usr/bin/env python3
"""
Combine individual disease type boxplots into a single vertical figure for Figure S2
"""

import os
from PIL import Image

def combine_plots_vertically(output_dir, preprocessing='CLR', output_filename=None):
    """
    Combine all disease type boxplots vertically into one large figure

    Args:
        output_dir: Directory containing individual plot files
        preprocessing: Preprocessing method used (default: 'CLR')
        output_filename: Name of combined output file (default: auto-generated)
    """
    # Disease types in desired order
    disease_types = ['Autoimmun', 'Intestinal', 'Liver', 'Mental', 'Metabolic']

    # Auto-generate output filename if not provided
    if output_filename is None:
        output_filename = f'combined_batchcorrection_boxplot_{preprocessing}.png'

    # Load all images
    images = []
    for disease_type in disease_types:
        img_path = os.path.join(output_dir, f'{disease_type}_batchcorrection_boxplot_{preprocessing}.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
            print(f"Loaded: {disease_type}_batchcorrection_boxplot_{preprocessing}.png ({img.size[0]}x{img.size[1]})")
        else:
            print(f"Warning: {img_path} not found, skipping...")

    if not images:
        print("No images found to combine!")
        return

    # Get dimensions
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]

    # Use maximum width to ensure alignment
    max_width = max(widths)
    total_height = sum(heights)

    print(f"\nCombining {len(images)} images...")
    print(f"Combined dimensions: {max_width}x{total_height}")

    # Create new image with white background
    combined = Image.new('RGB', (max_width, total_height), 'white')

    # Paste images vertically
    y_offset = 0
    for i, img in enumerate(images):
        # Center horizontally if image is narrower than max_width
        x_offset = (max_width - img.size[0]) // 2
        combined.paste(img, (x_offset, y_offset))
        print(f"Pasted {disease_types[i]} at y={y_offset}")
        y_offset += img.size[1]

    # Save combined image
    output_path = os.path.join(output_dir, output_filename)
    combined.save(output_path, dpi=(300, 300))
    print(f"\nSaved combined figure: {output_path}")
    print(f"Final size: {combined.size[0]}x{combined.size[1]} pixels")

def main():
    """Main function"""
    output_dir = '/ua/jmu27/Micro_bench/figures/figureS2'
    preprocessing = 'log_std'  # Can be changed to 'ALR' or 'CLR' if needed

    print("=" * 80)
    print(f"Combining Disease Type Boxplots ({preprocessing} preprocessing)")
    print("=" * 80)

    combine_plots_vertically(output_dir, preprocessing=preprocessing)

    print("\n" + "=" * 80)
    print("Combined figure generated successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()

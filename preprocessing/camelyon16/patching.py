import os
from glob import glob
import openslide
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import yaml
import argparse
import pandas as pd
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
import sys

def setup_logging(log_dir):
    """
    Set up logging to file and console.
    
    Args:
        log_dir (str): Directory to save log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "patching.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_xml(file_path):
    """
    Parse XML file and return root element.
    
    Args:
        file_path (str): Path to XML file.
    
    Returns:
        Element: Root element of the XML, or None if parsing fails.
    """
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return None

def extract_coordinates(file_path):
    """
    Extract tissue coordinates from XML file.
    
    Args:
        file_path (str): Path to XML file.
    
    Returns:
        list: List of (x, y) coordinates forming tissue polygons.
    """
    coordinates = []
    root = parse_xml(file_path)
    if root is None:
        return coordinates
    
    for coordinate in root.findall(".//Coordinate"):
        order = coordinate.attrib.get("Order")
        x = coordinate.attrib.get("X")
        y = coordinate.attrib.get("Y")
        if order and x and y:
            try:
                coordinates.append((float(x), float(y)))
            except ValueError:
                logging.warning(f"Invalid coordinate in {file_path}: x={x}, y={y}")
    
    # Sort by Order (assuming sequential connection)
    coordinates.sort(key=lambda c: c[0])  # Simplified; assumes Order is implicit
    return coordinates

def is_patch_in_tissue(coordinates, x_start, y_start, patch_size, tissue_threshold=0.1):
    """
    Check if a patch contains sufficient tissue based on XML coordinates.
    
    Args:
        coordinates (list): List of (x, y) coordinates forming tissue polygon.
        x_start (int): Top-left x-coordinate of the patch.
        y_start (int): Top-left y-coordinate of the patch.
        patch_size (int): Size of the patch (square).
        tissue_threshold (float): Minimum fraction of tissue pixels to keep patch.
    
    Returns:
        bool: True if patch has sufficient tissue, False otherwise.
    """
    if not coordinates:
        return False
    
    # Create a small mask for the patch region
    mask = Image.new("L", (patch_size, patch_size), 0)
    draw = ImageDraw.Draw(mask)
    
    # Adjust coordinates relative to patch
    relative_coords = [(x - x_start, y - y_start) for x, y in coordinates]
    
    # Draw polygon (clip to patch bounds)
    draw.polygon(relative_coords, fill=255)
    mask_array = np.array(mask)
    
    total_pixels = mask_array.size
    tissue_pixels = np.sum(mask_array > 0)
    tissue_fraction = tissue_pixels / total_pixels
    return tissue_fraction >= tissue_threshold

def extract_and_save_patches(
    wsi_dir,
    mask_dir,
    patch_save_dir,
    mask_save_dir,
    meta_save_dir,
    patch_size=256,
    stride=256,
    tissue_threshold=0.1
):
    """
    Extract patches from WSIs, filter by tissue contour (if mask exists), and save metadata.
    
    Args:
        wsi_dir (str): Directory containing WSI TIFFs.
        mask_dir (str): Directory containing XML mask files.
        patch_save_dir (str): Directory to save patches.
        mask_save_dir (str): Directory to save mask patches.
        meta_save_dir (str): Directory to save metadata.
        patch_size (int): Size of each patch (square).
        stride (int): Stride for patch extraction (default: patch_size for no overlap).
        tissue_threshold (float): Minimum tissue fraction to keep a patch.
    """
    # Create output directories
    patch_save_dir = Path(patch_save_dir)
    mask_save_dir = Path(mask_save_dir)
    meta_save_dir = Path(meta_save_dir)
    
    patch_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    meta_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get WSI image files
    image_paths = sorted(glob(os.path.join(wsi_dir, "*.tif")))
    logging.info(f"Found {len(image_paths)} WSI images")
    
    # Process each WSI
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing WSIs"), 1):
        slide_name = Path(img_path).stem
        logging.info(f"Processing {img_path}")
        
        # Get corresponding mask path
        mask_path = os.path.join(mask_dir, f"{slide_name}.xml")
        has_mask = os.path.exists(mask_path)
        
        # Load WSI
        try:
            slide = openslide.OpenSlide(img_path)
        except openslide.OpenSlideError as e:
            logging.error(f"Failed to open {img_path}: {e}")
            continue
        
        # Get dimensions at level 0
        w, h = slide.level_dimensions[0]
        
        # Validate dimensions
        if w * h > 10**10:  # Arbitrary limit (e.g., 100k x 100k)
            logging.warning(f"Skipping {img_path}: Image too large ({w}x{h})")
            slide.close()
            continue
        
        # Load mask coordinates if available
        mask_coordinates = []
        if has_mask:
            mask_coordinates = extract_coordinates(mask_path)
        
        # Calculate patch grid
        x_steps = (h - patch_size) // stride + 1
        y_steps = (w - patch_size) // stride + 1
        total_patches = x_steps * y_steps
        
        # Initialize metadata
        metadata = []
        
        # Progress bar for this WSI
        with tqdm(total=total_patches, desc=f"Extracting patches for {slide_name}", leave=False) as pbar:
            # Extract patches
            for xi in range(x_steps):
                for yi in range(y_steps):
                    x_start = xi * stride
                    y_start = yi * stride
                    
                    # Filter by tissue content if mask exists
                    if has_mask:
                        if not is_patch_in_tissue(mask_coordinates, x_start, y_start, patch_size, tissue_threshold):
                            pbar.update(1)
                            continue
                    
                    # Extract image patch
                    try:
                        img_patch = slide.read_region((y_start, x_start), 0, (patch_size, patch_size))
                        img_patch = img_patch.convert("RGB")
                        img_patch_array = np.array(img_patch)
                    except openslide.OpenSlideError as e:
                        logging.warning(f"Failed to read patch at ({x_start}, {y_start}) in {img_path}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Generate patch ID
                    patch_id = str(uuid.uuid4())
                    
                    # Save image patch
                    patch_filename = f"{slide_name}_{patch_id}_{x_start}_{y_start}.png"
                    patch_path = patch_save_dir / patch_filename
                    try:
                        img_patch.save(patch_path)
                    except Exception as e:
                        logging.warning(f"Failed to save patch {patch_filename}: {e}")
                        pbar.update(1)
                        continue
                    
                    # Save mask patch if available
                    mask_filename = None
                    mask_path = None
                    if has_mask:
                        # Generate mask patch for this region
                        mask_patch = Image.new("L", (patch_size, patch_size), 0)
                        draw = ImageDraw.Draw(mask_patch)
                        relative_coords = [(x - x_start, y - y_start) for x, y in mask_coordinates]
                        draw.polygon(relative_coords, fill=255)
                        mask_patch_array = np.array(mask_patch)
                        
                        mask_filename = f"{slide_name}_{patch_id}_{x_start}_{y_start}_mask.png"
                        mask_path = mask_save_dir / mask_filename
                        try:
                            mask_patch.save(mask_path)
                        except Exception as e:
                            logging.warning(f"Failed to save mask patch {mask_filename}: {e}")
                    
                    # Store metadata
                    metadata.append({
                        "patch_id": patch_id,
                        "slide_name": slide_name,
                        "patch_filename": patch_filename,
                        "mask_filename": mask_filename or "",
                        "x_start": x_start,
                        "y_start": y_start,
                        "patch_size": patch_size,
                        "stride": stride,
                        "patch_path": str(patch_path),
                        "mask_path": str(mask_path) if mask_path else "",
                        "has_mask": has_mask
                    })
                    
                    pbar.update(1)
        
        # Save metadata to CSV
        try:
            metadata_df = pd.DataFrame(metadata)
            metadata_csv = meta_save_dir / f"{slide_name}_metadata.csv"
            metadata_df.to_csv(metadata_csv, index=False)
            logging.info(f"Processed {img_path}: {len(metadata)} patches saved, metadata at {metadata_csv}")
        except Exception as e:
            logging.error(f"Failed to save metadata for {img_path}: {e}")
        
        # Clean up
        slide.close()
        del slide
        if has_mask:
            del mask_coordinates

def main():
    dataset_name = "camelyon16"
    parser = argparse.ArgumentParser(description="Extract patches from WSI TIFFs with XML mask filtering")
    parser.add_argument("--config", type=str, default=f"configs/data_{dataset_name}.yaml", help="Path to YAML config file")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of each patch")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch extraction")
    parser.add_argument("--tissue_threshold", type=float, default=0.1, help="Minimum tissue fraction to keep patch")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config {args.config}: {e}")
        sys.exit(1)
    
    # Validate config
    required_keys = ["WSI_DIR", "WSI_MASK_DIR", "PATCH_DIR", "PATCH_MASK_DIR", "PATCH_META_DIR", "LOG_DIR"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing config keys: {missing_keys}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config["LOG_DIR"])
    
    # Extract patches
    extract_and_save_patches(
        config["WSI_DIR"],
        config["WSI_MASK_DIR"],
        config["PATCH_DIR"],
        config["PATCH_MASK_DIR"],
        config["PATCH_META_DIR"],
        patch_size=args.patch_size,
        stride=args.stride,
        tissue_threshold=args.tissue_threshold
    )

if __name__ == "__main__":
    main()
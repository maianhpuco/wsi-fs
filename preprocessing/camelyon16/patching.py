import os
from glob import glob
import large_image
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
import psutil
import gc

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

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")

def parse_xml(file_path):
    """
    Parse XML file and return root element.
    
    Args:
        file_path (str): Path to XML file.
    
    Returns:
        Element: Root element of the XML, or None if parsing fails.
    """
    try:
        tree = ET.iterparse(file_path, events=("start", "end"))
        root = None
        for event, elem in tree:
            if event == "start" and root is None:
                root = elem
            if event == "end":
                elem.clear()  # Free memory
        return root
    except ET.ParseError as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return None

def extract_coordinates(file_path, scale_factor=1.0):
    """
    Extract tissue coordinates from XML file, scaled by factor.
    
    Args:
        file_path (str): Path to XML file.
        scale_factor (float): Factor to scale coordinates (e.g., 0.5 for level 1).
    
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
                coordinates.append((float(x) * scale_factor, float(y) * scale_factor))
            except ValueError:
                logging.warning(f"Invalid coordinate in {file_path}: x={x}, y={y}")
    
    # Sort by Order (simplified)
    coordinates.sort(key=lambda c: c[0])
    logging.info(f"Extracted {len(coordinates)} coordinates from {file_path}")
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
    mask.close()
    
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
    tissue_threshold=0.1,
    level=1
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
        stride (int): Stride for patch extraction.
        tissue_threshold (float): Minimum tissue fraction to keep a patch.
        level (int): Resolution level to process (0=highest, 1=2x downsampled, etc.).
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
        log_memory_usage()
        
        # Get corresponding mask path
        mask_path = os.path.join(mask_dir, f"{slide_name}.xml")
        has_mask = os.path.exists(mask_path)
        
        # Load WSI
        try:
            slide = large_image.getTileSource(img_path)
        except Exception as e:
            logging.error(f"Failed to open {img_path}: {e}")
            continue
        
        # Get dimensions at specified level
        level_info = slide.getLevelForMagnification(slide.getNativeMagnification()['magnification'] / (2 ** level))
        w, h = slide.getDimensions(level=level_info)
        scale_factor = slide.getMagnificationForLevel(level) / slide.getNativeMagnification()['magnification']
        
        logging.info(f"Processing at level {level}: {w}x{h}, scale factor {scale_factor:.4f}")
        
        # Load mask coordinates if available
        mask_coordinates = []
        if has_mask:
            mask_coordinates = extract_coordinates(mask_path, scale_factor)
        
        # Calculate patch grid
        x_steps = (h - patch_size) // stride + 1
        y_steps = (w - patch_size) // stride + 1
        total_patches = x_steps * y_steps
        batch_size = 1000  # Save metadata every 1000 patches
        
        # Initialize metadata
        metadata = []
        
        # Progress bar for this WSI
        with tqdm(total=total_patches, desc=f"Extracting patches for {slide_name}", leave=False, mininterval=1.0) as pbar:
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
                        tile, _ = slide.getSingleTile(
                            tile_position=(yi * stride, xi * stride),
                            tile_size=(patch_size, patch_size),
                            level=level_info
                        )
                        img_patch = Image.frombytes("RGB", (patch_size, patch_size), tile['data'])
                        img_patch_array = np.array(img_patch)
                    except Exception as e:
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
                        img_patch.close()
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
                            mask_patch.close()
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
                        "has_mask": has_mask,
                        "level": level
                    })
                    
                    # Save metadata incrementally
                    if len(metadata) >= batch_size:
                        try:
                            metadata_df = pd.DataFrame(metadata)
                            metadata_csv = meta_save_dir / f"{slide_name}_metadata.csv"
                            metadata_df.to_csv(metadata_csv, index=False, mode="a", header=not metadata_csv.exists())
                            metadata = []
                            gc.collect()
                        except Exception as e:
                            logging.error(f"Failed to save metadata batch for {img_path}: {e}")
                    
                    pbar.update(1)
        
        # Save remaining metadata
        if metadata:
            try:
                metadata_df = pd.DataFrame(metadata)
                metadata_csv = meta_save_dir / f"{slide_name}_metadata.csv"
                metadata_df.to_csv(metadata_csv, index=False, mode="a", header=not metadata_csv.exists())
                logging.info(f"Processed {img_path}: {len(metadata)} patches saved, metadata at {metadata_csv}")
            except Exception as e:
                logging.error(f"Failed to save final metadata for {img_path}: {e}")
        
        # Clean up
        del slide
        del mask_coordinates
        gc.collect()
        log_memory_usage()

def main():
    dataset_name = "camelyon16"
    parser = argparse.ArgumentParser(description="Extract patches from WSI TIFFs with XML mask filtering")
    parser.add_argument("--config", type=str, default=f"configs/data_{dataset_name}.yaml", help="Path to YAML config file")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of each patch")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch extraction")
    parser.add_argument("--tissue_threshold", type=float, default=0.1, help="Minimum tissue fraction to keep patch")
    parser.add_argument("--level", type=int, default=1, help="Resolution level to process (0=highest)")
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
        tissue_threshold=args.tissue_threshold,
        level=args.level
    )

if __name__ == "__main__":
    main()
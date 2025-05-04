import os
from glob import glob
import tifffile
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import yaml
import argparse
import pandas as pd
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET

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
        print(f"Error parsing {file_path}: {e}")
        return None
    
def reset_dir(path: Path):
    import shutil 
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True) 
    
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
            coordinates.append((float(x), float(y)))
    
    # Sort by Order (assuming sequential connection)
    coordinates.sort(key=lambda c: c[0])  # Simplified; assumes Order is implicit
    return coordinates

def generate_mask(wsi_shape, xml_path):
    """
    Generate binary mask from XML coordinates.
    
    Args:
        wsi_shape (tuple): Height and width of the WSI.
        xml_path (str): Path to XML file with tissue coordinates.
    
    Returns:
        np.ndarray: Binary mask (0=background, 255=tissue).
    """
    height, width = wsi_shape[:2]
    mask = Image.new("L", (width, height), 0)  # Black background
    draw = ImageDraw.Draw(mask)
    
    coordinates = extract_coordinates(xml_path)
    if not coordinates:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Draw tissue polygon (assuming closed polygon)
    draw.polygon(coordinates, fill=255)  # White tissue
    return np.array(mask)

def filter_patch_by_contour(mask_patch, tissue_threshold=0.1):
    """
    Filter patches based on tissue content in the mask.
    
    Args:
        mask_patch (np.ndarray): Binary mask patch (0=background, non-zero=tissue).
        tissue_threshold (float): Minimum fraction of tissue pixels to keep patch.
    
    Returns:
        bool: True if patch has sufficient tissue, False otherwise.
    """
    total_pixels = mask_patch.size
    tissue_pixels = np.sum(mask_patch > 0)
    tissue_fraction = tissue_pixels / total_pixels
    return tissue_fraction >= tissue_threshold

def extract_and_save_patches(
    wsi_dir, 
    mask_dir, 
    patch_save_dir, 
    mask_save_dir, 
    meta_save_dir, 
    patch_size=256, stride=256, tissue_threshold=0.1):
    """
    Extract patches from WSIs, filter by tissue contour (if mask exists), and save metadata.
    
    Args:
        wsi_dir (str): Directory containing WSI TIFFs.
        mask_dir (str): Directory containing XML mask files.
        patch_save_dir (str): Directory to save patches and metadata.
        mask_save_dir (str): Directory to save mask patches. 
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
    # ---------- reset directories if needed ---------- 
    # reset_dir(patch_save_dir)
    # reset_dir(mask_save_dir)
    # reset_dir(meta_save_dir) 
    # Get WSI image files
    image_paths = sorted(glob(os.path.join(wsi_dir, "*.tif")))
    print(f"Found {len(image_paths)} WSI images")
    
    # Process each WSI
    for img_path in tqdm(image_paths, desc="Processing WSIs"):
        # Get corresponding mask path
        slide_name = Path(img_path).stem
        mask_path = os.path.join(mask_dir, f"{slide_name}.xml")
        has_mask = os.path.exists(mask_path)
        
        # Load WSI
        wsi_img = tifffile.imread(img_path)
        
        # Ensure RGB
        if wsi_img.ndim == 2:
            wsi_img = np.stack([wsi_img] * 3, axis=-1)
        
        # Get dimensions
        h, w = wsi_img.shape[:2]
        
        # Generate mask if available
        wsi_mask = None
        if has_mask:
            wsi_mask = generate_mask(wsi_img.shape, mask_path)
        
        # Calculate patch grid
        x_steps = (h - patch_size) // stride + 1
        y_steps = (w - patch_size) // stride + 1
        
        # Initialize metadata
        metadata = []
        
        # Extract patches
        for xi in range(x_steps):
            for yi in range(y_steps):
                x_start = xi * stride
                y_start = yi * stride
                
                # Extract image patch
                img_patch = wsi_img[x_start:x_start + patch_size, y_start:y_start + patch_size]
                
                # Filter by tissue content if mask exists
                if has_mask:
                    mask_patch = wsi_mask[x_start:x_start + patch_size, y_start:y_start + patch_size]
                    if not filter_patch_by_contour(mask_patch, tissue_threshold):
                        continue
                
                # Generate patch ID
                patch_id = str(uuid.uuid4())
                
                # Save image patch
                patch_filename = f"{slide_name}_{patch_id}_{x_start}_{y_start}.png"
                patch_path = patch_save_dir / patch_filename
                Image.fromarray(img_patch).save(patch_path)
                
                # Save mask patch if available
                mask_filename = None
                mask_path = None
                if has_mask:
                    mask_filename = f"{slide_name}_{patch_id}_{x_start}_{y_start}_mask.png"
                    mask_path = mask_save_dir / mask_filename
                    Image.fromarray(mask_patch).save(mask_path)
                
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
        
        # Save metadata to CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_csv = meta_save_dir / f"{slide_name}_metadata.csv"
        metadata_df.to_csv(metadata_csv, index=False)
        
        print(f"Processed {img_path}: {len(metadata)} patches saved, metadata at {metadata_csv}")

def main():
    dataset_name = "camelyon16" 
    parser = argparse.ArgumentParser(description="Extract patches from WSI TIFFs with XML mask filtering")
    parser.add_argument("--config", type=str, default=f"config/data_{dataset_name}.yaml", help="Path to YAML config file")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of each patch")
    parser.add_argument("--stride", type=int, default=256, help="Stride for patch extraction")
    parser.add_argument("--tissue_threshold", type=float, default=0.1, help="Minimum tissue fraction to keep patch")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure output directories exist
    
    Path(config["PATCH_MASK_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["PATH_META_DIR"]).mkdir(parents=True, exist_ok=True)
  
    # Extract patches
    extract_and_save_patches(
        config["WSI_DIR"], 
        config["WSI_MASK_DIR"],
        config["PATCH_DIR"],   
        config["PATCH_MASK_DIR"], 
        config["PATH_META_DIR"], 
        patch_size=args.patch_size,
        stride=args.stride,
        tissue_threshold=args.tissue_threshold
    )

if __name__ == "__main__":
    main()
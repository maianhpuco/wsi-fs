"""
Patching module for WSI processing.

This module provides a function to extract patches from Whole Slide Images (WSIs)
and save them as HDF5 and PNG files.
"""

import os
import sys
import time
import logging
from typing import Dict, float
from PIL import Image

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def patch_wsi(
    wsi: WholeSlideImage,
    patch_params: Dict,
    patch_size: int,
    step_size: int,
    patch_level: int,
    h5_path: str,
    patch_png_dir: str
) -> float:
    """Extract patches from the WSI and save as HDF5 and PNG files.

    Args:
        wsi: WholeSlideImage object to patch.
        patch_params: Parameters for patch extraction.
        patch_size: Size of each patch (e.g., 256).
        step_size: Step size for patch extraction.
        patch_level: Downsample level for patching.
        h5_path: Path to save the HDF5 file.
        patch_png_dir: Directory to save PNG patches.

    Returns:
        Time taken for patching (in seconds).
    """
    start_time = time.time()
    
    # Prepare patch parameters for process_contours, excluding patch_png_dir
    contour_params = patch_params.copy()
    contour_params.update({
        'patch_level': patch_level,
        'patch_size': patch_size,
        'step_size': step_size,
        'save_path': h5_path
    })
    if 'patch_png_dir' in contour_params:
        del contour_params['patch_png_dir']  # Remove to avoid passing to process_contours
    
    magnification = wsi.wsi.properties['aperio.AppMag']
    contour_params['mag'] = str(magnification)
    
    # Call process_contours to extract patches (assumes it returns patches and coordinates)
    try:
        patch_data = wsi.process_contours(**contour_params)  # Expecting list of (patch_image, coord) tuples
    except TypeError as e:
        logger.error(f"Error in process_contours: {e}")
        raise

    # Save PNG patches if patch_data contains images
    os.makedirs(patch_png_dir, exist_ok=True)
    if patch_data and isinstance(patch_data, (list, tuple)):
        for idx, item in enumerate(patch_data):
            if isinstance(item, tuple) and len(item) >= 2:
                patch_image, coord = item[:2]
                if isinstance(patch_image, Image.Image):
                    patch_filename = os.path.join(patch_png_dir, f"patch_{coord[0]}_{coord[1]}.png")
                    patch_image.save(patch_filename)
                    logger.info(f"Saved PNG patch: {patch_filename}")
                else:
                    logger.warning(f"Patch at index {idx} is not a PIL Image: {type(patch_image)}")
            else:
                logger.warning(f"Unexpected patch data format at index {idx}: {item}")
    else:
        logger.warning("No patch data returned by process_contours, skipping PNG saving")

    return time.time() - start_time
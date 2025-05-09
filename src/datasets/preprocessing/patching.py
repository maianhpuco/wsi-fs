"""
Patching module for WSI processing.

This module provides a function to extract patches from Whole Slide Images (WSIs)
and save them as HDF5 and PNG files.
"""

import os
import sys
import time
from typing import Dict, float

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage

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
    patch_params.update({
        'patch_level': patch_level,
        'patch_size': patch_size,
        'step_size': step_size,
        'save_path': h5_path,
        'patch_png_dir': patch_png_dir  # Custom parameter for PNG saving
    })
    magnification = wsi.wsi.properties['aperio.AppMag']
    patch_params['mag'] = str(magnification)
    wsi.process_contours(**patch_params)  # Assumes process_contours handles both HDF5 and PNG
    return time.time() - start_time
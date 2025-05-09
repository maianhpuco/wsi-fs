"""
Patching module for WSI processing.

This module provides a function to extract patches from Whole Slide Images (WSIs)
based on specified parameters, saving them to an HDF5 file.
"""

import os
import sys
import time
from typing import Dict, Tuple

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage

def patch_wsi(wsi: WholeSlideImage, patch_params: Dict, patch_size: int, step_size: int, patch_level: int, save_path: str) -> Tuple[str, float]:
    """Extract patches from the WSI.

    Args:
        wsi: WholeSlideImage object to patch.
        patch_params: Parameters for patch extraction.
        patch_size: Size of each patch (e.g., 256).
        step_size: Step size for patch extraction.
        patch_level: Downsample level for patching.
        save_path: Directory to save the patch HDF5 file.

    Returns:
        Tuple containing the path to the saved HDF5 file and the time taken.
    """
    start_time = time.time()
    patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'save_path': save_path})
    magnification = wsi.wsi.properties['aperio.AppMag']
    patch_params['mag'] = str(magnification)
    file_path = wsi.process_contours(**patch_params)
    return file_path, time.time() - start_time
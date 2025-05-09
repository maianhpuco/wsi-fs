"""
patching module for wsi processing.

this module provides a function to extract patches from whole slide images (wsis)
and save them as hdf5 and png files.
"""

import os
import sys
import time
from typing import dict, float

# set project root for importing custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.datasets.wsi_core.wholeslideimage import wholeslideimage

def patch_wsi(
    wsi: wholeslideimage,
    patch_params: dict,
    patch_size: int,
    step_size: int,
    patch_level: int,
    h5_path: str,
    patch_png_dir: str
) -> float:
    """extract patches from the wsi and save as hdf5 and png files.

    args:
        wsi: wholeslideimage object to patch.
        patch_params: parameters for patch extraction.
        patch_size: size of each patch (e.g., 256).
        step_size: step size for patch extraction.
        patch_level: downsample level for patching.
        h5_path: path to save the hdf5 file.
        patch_png_dir: directory to save png patches.

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
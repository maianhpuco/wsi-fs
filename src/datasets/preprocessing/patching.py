"""
patching module for wsi processing.

this module provides a function to extract patches from whole slide images (wsis)
based on specified parameters, saving them to an hdf5 file.
"""

import os
import sys
import time
from typing import Dict, Tuple


# set project root for importing custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.datasets.wsi_core.WholeSlideImage import wholeslideimage

def patch_wsi(wsi: wholeslideimage, patch_params: Dict, patch_size: int, step_size: int, patch_level: int, save_path: str) -> Tuple[str, float]:
    """extract patches from the wsi.

    args:
        wsi: wholeslideimage object to patch.
        patch_params: parameters for patch extraction.
        patch_size: size of each patch (e.g., 256).
        step_size: step size for patch extraction.
        patch_level: downsample level for patching.
        save_path: directory to save the patch hdf5 file.

    returns:
        Tuple containing the path to the saved hdf5 file and the time taken.
    """
    start_time = time.time()
    patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'save_path': save_path})
    magnification = wsi.wsi.properties['aperio.appmag']
    patch_params['mag'] = str(magnification)
    file_path = wsi.process_contours(**patch_params)
    return file_path, time.time() - start_time
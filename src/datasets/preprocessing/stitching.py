"""
Stitching module for WSI processing.

This module provides a function to stitch patch coordinates from an HDF5 file
into a heatmap for visualization.
"""

import os
import sys
import time
from typing import Tuple

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage
from src.datasets.wsi_core.wsi_utils import StitchCoords

def stitch_wsi(file_path: str, wsi: WholeSlideImage, downscale: int = 64) -> Tuple[any, float]:
    """Stitch patch coordinates into a heatmap.

    Args:
        file_path: Path to the HDF5 file containing patch coordinates.
        wsi: WholeSlideImage object for the slide.
        downscale: Downscale factor for the heatmap.

    Returns:
        Tuple containing the stitched heatmap and the time taken.
    """
    start_time = time.time()
    heatmap = StitchCoords(file_path, wsi, downscale=downscale, bg_color=(255, 255, 255), alpha=-1, draw_grid=True)
    return heatmap, time.time() - start_time
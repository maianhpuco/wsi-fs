"""
Segmentation module for WSI processing.

This module provides functions to segment tissue in Whole Slide Images (WSIs) and
update segmentation parameters for each slide, including legacy support for older
CSV formats.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage
from config import ProcessingConfig

def update_params_for_slide(df: pd.DataFrame, idx: int, wsi: WholeSlideImage, config: ProcessingConfig, legacy_support: bool) -> Tuple[Dict, Dict, Dict, Dict]:
    """Update parameters for a specific slide, handling legacy support.

    Args:
        df: DataFrame containing processing parameters.
        idx: Index of the slide in the DataFrame.
        wsi: WholeSlideImage object for the slide.
        config: Configuration object with default parameters.
        legacy_support: Whether to handle legacy CSV formats.

    Returns:
        Tuple of updated parameter dictionaries for segmentation, filtering,
        visualization, and patching.
    """
    current_seg_params = config.seg_params.copy()
    current_filter_params = config.filter_params.copy()
    current_vis_params = config.vis_params.copy()
    current_patch_params = config.patch_params.copy()

    if not config.use_default_params:
        for key in current_vis_params:
            if legacy_support and key == 'vis_level':
                df.loc[idx, key] = -1
            current_vis_params[key] = df.loc[idx, key]

        for key in current_filter_params:
            if legacy_support and key == 'a_t':
                old_area = df.loc[idx, 'a']
                seg_level = df.loc[idx, 'seg_level']
                scale = wsi.level_downsamples[seg_level]
                adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                current_filter_params[key] = adjusted_area
                df.loc[idx, key] = adjusted_area
            else:
                current_filter_params[key] = df.loc[idx, key]

        for key in current_seg_params:
            if legacy_support and key == 'seg_level':
                df.loc[idx, key] = -1
            current_seg_params[key] = df.loc[idx, key]

        for key in current_patch_params:
            current_patch_params[key] = df.loc[idx, key]

    # Adjust visualization and segmentation levels
    for param, key in [(current_vis_params, 'vis_level'), (current_seg_params, 'seg_level')]:
        if param[key] < 0:
            param[key] = 0 if len(wsi.level_dim) == 1 else wsi.getOpenSlide().get_best_level_for_downsample(64)

    # Handle keep/exclude IDs
    for key in ['keep_ids', 'exclude_ids']:
        ids_str = str(current_seg_params[key])
        current_seg_params[key] = np.array(ids_str.split(','), dtype=int) if ids_str != 'none' and len(ids_str) > 0 else []

    return current_seg_params, current_filter_params, current_vis_params, current_patch_params

def segment_wsi(wsi: WholeSlideImage, seg_params: Dict, filter_params: Dict) -> Tuple[WholeSlideImage, float]:
    """Segment tissue in the WSI.

    Args:
        wsi: WholeSlideImage object to segment.
        seg_params: Parameters for segmentation.
        filter_params: Parameters for filtering segmented regions.

    Returns:
        Tuple containing the segmented WSI and the time taken for segmentation.
    """
    start_time = time.time()
    wsi.segmentTissue(**seg_params, filter_params=filter_params)
    return wsi, time.time() - start_time
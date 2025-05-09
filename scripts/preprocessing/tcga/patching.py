"""
Entry point script for TCGA WSI processing pipeline.

This script runs the WSI processing pipeline for TCGA datasets (e.g., KICH), including
loading slides, segmenting tissue, extracting patches, and stitching heatmaps. It loads
configuration from a YAML file specified via command-line arguments.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import inspect

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
print(f"Project root set to: {PROJECT_ROOT}")

from src.datasets.preprocessing.config import ProcessingConfig, load_config
from src.datasets.preprocessing.slide_loader import load_slides
from src.datasets.preprocessing.segmentation import update_params_for_slide, segment_wsi
from src.datasets.preprocessing.patching import patch_wsi
from src.datasets.preprocessing.stitching import stitch_wsi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args() -> str:
    """Parse command-line arguments for the configuration file path.

    Returns:
        Path to the YAML configuration file.
    """
    parser = argparse.ArgumentParser(description='Run TCGA WSI processing pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    return args.config

def process_slide(idx: int, slide: str, df: pd.DataFrame, id_names: Dict[str, str], config: ProcessingConfig, legacy_support: bool) -> Tuple[float, float, float]:
    """Process a single slide (segmentation, patching, stitching).

    Args:
        idx: Index of the slide in the DataFrame.
        slide: Filename of the slide.
        df: DataFrame containing processing parameters.
        id_names: Dictionary mapping slide filenames to UUIDs.
        config: Configuration object with processing parameters.
        legacy_support: Whether to handle legacy CSV formats.

    Returns:
        Tuple of times taken for segmentation, patching, and stitching.
    """
    slide_id, _ = os.path.splitext(slide)
    df.loc[idx, 'process'] = 0

    if config.auto_skip and os.path.isfile(os.path.join(config.patch_h5_dir, f"{slide_id}.h5")):
        logger.info(f"{slide_id} already exists, skipping")
        df.loc[idx, 'status'] = 'already_exist'
        return 0.0, 0.0, 0.0

    full_path = os.path.join(config.source_dir, id_names[slide], slide)
    from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage
    wsi = WholeSlideImage(full_path)

    seg_params, filter_params, vis_params, patch_params = update_params_for_slide(df, idx, wsi, config, legacy_support)

    w, h = wsi.level_dim[seg_params['seg_level']]
    if w * h > 1e8:
        logger.error(f"Image too large for segmentation ({w}x{h}), aborting")
        df.loc[idx, 'status'] = 'failed_seg'
        return 0.0, 0.0, 0.0

    df.loc[idx, 'vis_level'] = vis_params['vis_level']
    df.loc[idx, 'seg_level'] = seg_params['seg_level']

    seg_time, patch_time, stitch_time = 0.0, 0.0, 0.0
    if config.seg:
        wsi, seg_time = segment_wsi(wsi, seg_params, filter_params)

    if config.save_mask:
        mask, only_mask = wsi.visWSI(**vis_params)
        mask.save(os.path.join(config.mask_save_dir, f"{slide_id}.jpg"))
        only_mask.save(os.path.join(config.only_mask_save_dir, f"{slide_id}.png"))

    if config.patch:
        h5_path = os.path.join(config.patch_h5_dir, f"{slide_id}.h5")
        patch_png_dir = os.path.join(config.patch_png_dir, slide_id)
        os.makedirs(patch_png_dir, exist_ok=True)
        try:
            patch_time = patch_wsi(
                wsi=wsi,
                patch_params=patch_params,
                patch_size=config.patch_size,
                step_size=config.step_size,
                patch_level=config.patch_level,
                h5_path=h5_path,
                patch_png_dir=patch_png_dir
            )
        except TypeError as e:
            logger.error(f"TypeError in patch_wsi: {e}")
            logger.error(f"patch_wsi signature: {inspect.signature(patch_wsi)}")
            raise

    if config.stitch and os.path.isfile(os.path.join(config.patch_h5_dir, f"{slide_id}.h5")):
        heatmap, stitch_time = stitch_wsi(os.path.join(config.patch_h5_dir, f"{slide_id}.h5"), wsi)
        heatmap.save(os.path.join(config.stitch_save_dir, f"{slide_id}.jpg"))

    logger.info(f"Segmentation: {seg_time:.2f}s, Patching: {patch_time:.2f}s, Stitching: {stitch_time:.2f}s")
    df.loc[idx, 'status'] = 'processed'
    return seg_time, patch_time, stitch_time

def main(config: ProcessingConfig) -> Tuple[float, float]:
    """Main function to process WSIs.

    Args:
        config: Configuration object with processing parameters.

    Returns:
        Tuple of average segmentation and patching times.
    """
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.patch_h5_dir, exist_ok=True)
    os.makedirs(config.patch_png_dir, exist_ok=True)
    os.makedirs(config.mask_save_dir, exist_ok=True)
    os.makedirs(config.only_mask_save_dir, exist_ok=True)
    os.makedirs(config.stitch_save_dir, exist_ok=True)

    slides, id_names, df = load_slides(config)
    process_stack = df[df['process'] == 1]
    total = len(process_stack)
    legacy_support = 'a' in df.columns

    if legacy_support:
        logger.info("Detected legacy segmentation CSV, enabling legacy support")
        df = df.assign(**{
            'a_t': np.full(len(df), int(config.filter_params['a_t']), dtype=np.uint32),
            'a_h': np.full(len(df), int(config.filter_params['a_h']), dtype=np.uint32),
            'max_n_holes': np.full(len(df), int(config.filter_params['max_n_holes']), dtype=np.uint32),
            'line_thickness': np.full(len(df), int(config.vis_params['line_thickness']), dtype=np.uint32),
            'contour_fn': np.full(len(df), config.patch_params['contour_fn'])
        })

    seg_times, patch_times, stitch_times = 0.0, 0.0, 0.0
    for i, (idx, row) in enumerate(process_stack.iterrows()):
        logger.info(f"Progress: {i/total:.2%}, {i}/{total}, processing {row['slide_id']}")
        seg_time, patch_time, stitch_time = process_slide(idx, row['slide_id'], df, id_names, config, legacy_support)
        seg_times += seg_time
        patch_times += patch_time
        stitch_times += stitch_time
        df.to_csv(os.path.join(config.save_dir, 'process_list_autogen.csv'), index=False)

    avg_seg_time = seg_times / total if total > 0 else 0.0
    avg_patch_time = patch_times / total if total > 0 else 0.0
    avg_stitch_time = stitch_times / total if total > 0 else 0.0
    logger.info(f"Average segmentation time: {avg_seg_time:.2f}s")
    logger.info(f"Average patching time: {avg_patch_time:.2f}s")
    logger.info(f"Average stitching time: {avg_stitch_time:.2f}s")

    return avg_seg_time, avg_patch_time

if __name__ == '__main__':
    config_path = parse_args()
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    logger.info(f"Configuration: {config}")
    main(config)
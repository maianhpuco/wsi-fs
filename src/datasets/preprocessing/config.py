"""
Configuration module for WSI processing.

This module defines the `ProcessingConfig` dataclass to store parameters for WSI processing
and includes a function to parse command-line arguments. It sets default parameters for
segmentation, filtering, visualization, and patching, and supports preset configurations
from CSV files.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

@dataclass
class ProcessingConfig:
    """Configuration for WSI processing parameters."""
    source: str
    save_dir: str
    patch_save_dir: str
    mask_save_dir: str
    only_mask_save_dir: str
    stitch_save_dir: str
    slide_name_file: str
    uuid_name_file: str
    patch_size: int = 256
    step_size: int = 256
    patch_level: int = 0
    seg: bool = False
    patch: bool = False
    stitch: bool = False
    save_mask: bool = True
    auto_skip: bool = True
    process_list: Optional[str] = None
    seg_params: Dict = None
    filter_params: Dict = None
    vis_params: Dict = None
    patch_params: Dict = None
    use_default_params: bool = False

    def __post_init__(self):
        """Initialize default parameters if not provided."""
        self.seg_params = self.seg_params or {
            'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4,
            'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'
        }
        self.filter_params = self.filter_params or {
            'a_t': 100, 'a_h': 16, 'max_n_holes': 8
        }
        self.vis_params = self.vis_params or {
            'vis_level': -1, 'line_thickness': 250
        }
        self.patch_params = self.patch_params or {
            'use_padding': True, 'contour_fn': 'four_pt'
        }

def parse_args() -> ProcessingConfig:
    """Parse command-line arguments for WSI processing.

    Returns:
        ProcessingConfig: Configuration object with parsed parameters.
    """
    parser = argparse.ArgumentParser(description='Segment and patch WSIs')
    parser.add_argument('--source', type=str, required=True, help='Path to WSI image files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--slide_name_file', type=str, required=True, help='File with slide names')
    parser.add_argument('--uuid_name_file', type=str, required=True, help='File with slide info')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--step_size', type=int, default=256, help='Step size')
    parser.add_argument('--patch_level', type=int, default=0, help='Downsample level for patching')
    parser.add_argument('--seg', action='store_true', help='Enable segmentation')
    parser.add_argument('--patch', action='store_true', help='Enable patching')
    parser.add_argument('--stitch', action='store_true', help='Enable stitching')
    parser.add_argument('--no_auto_skip', action='store_false', help='Disable auto-skip for existing files')
    parser.add_argument('--process_list', type=str, default=None, help='CSV list of images to process')
    parser.add_argument('--preset', type=str, default=None, help='Preset CSV for parameters')

    args = parser.parse_args()
    
    patch_save_dir = os.path.join(args.save_dir, f'patches_{args.patch_size}')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    only_mask_save_dir = os.path.join(args.save_dir, 'only_masks')
    stitch_save_dir = os.path.join(args.save_dir, f'graph_{args.patch_size}')

    config = ProcessingConfig(
        source=args.source,
        save_dir=args.save_dir,
        patch_save_dir=patch_save_dir,
        mask_save_dir=mask_save_dir,
        only_mask_save_dir=only_mask_save_dir,
        stitch_save_dir=stitch_save_dir,
        slide_name_file=args.slide_name_file,
        uuid_name_file=args.uuid_name_file,
        patch_size=args.patch_size,
        step_size=args.step_size,
        patch_level=args.patch_level,
        seg=args.seg,
        patch=args.patch,
        stitch=args.stitch,
        auto_skip=args.no_auto_skip,
        process_list=args.process_list
    )

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for params in [config.seg_params, config.filter_params, config.vis_params, config.patch_params]:
            for key in params:
                params[key] = preset_df.loc[0, key]

    return config
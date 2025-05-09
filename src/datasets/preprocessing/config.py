"""
Configuration module for WSI processing.

This module defines the `ProcessingConfig` dataclass to store parameters for WSI processing
and includes a function to load configurations from a YAML file. It sets default parameters
for segmentation, filtering, visualization, and patching, and supports preset configurations
from CSV files.
"""

import os
import sys
import yaml
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

@dataclass
class ProcessingConfig:
    """Configuration for WSI processing parameters."""
    source_dir: str
    save_dir: str
    patch_h5_dir: str
    patch_png_dir: str
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

def load_config(config_path: str = "configs/wsi_processing.yaml") -> ProcessingConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ProcessingConfig: Configuration object with parameters from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Create ProcessingConfig object
    config = ProcessingConfig(
        source_dir=config_data['source_dir'],
        save_dir=config_data['save_dir'],
        patch_h5_dir=config_data['patch_h5_dir'],
        patch_png_dir=config_data['patch_png_dir'],
        mask_save_dir=config_data['mask_save_dir'],
        only_mask_save_dir=config_data['only_mask_save_dir'],
        stitch_save_dir=config_data['stitch_save_dir'],
        slide_name_file=config_data['slide_name_file'],
        uuid_name_file=config_data['uuid_name_file'],
        patch_size=config_data['patch_size'],
        step_size=config_data['step_size'],
        patch_level=config_data['patch_level'],
        seg=config_data['seg'],
        patch=config_data['patch'],
        stitch=config_data['stitch'],
        save_mask=config_data['save_mask'],
        auto_skip=config_data['auto_skip'],
        process_list=config_data['process_list'],
        seg_params=config_data['seg_params'],
        filter_params=config_data['filter_params'],
        vis_params=config_data['vis_params'],
        patch_params=config_data['patch_params'],
        use_default_params=config_data['use_default_params']
    )

    # Handle preset CSV if provided
    if config_data.get('preset'):
        preset_df = pd.read_csv(os.path.join('presets', config_data['preset']))
        for params in [config.seg_params, config.filter_params, config.vis_params, config.patch_params]:
            for key in params:
                params[key] = preset_df.loc[0, key]

    return config
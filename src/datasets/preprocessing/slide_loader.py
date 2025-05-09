"""
Slide loading module for WSI processing.

This module provides functions to load slide information from an Excel file and
initialize a DataFrame for processing WSIs. It maps slide names to their UUIDs
and filters out invalid slide paths.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.datasets.wsi_core.batch_process_utils import initialize_df
from config import ProcessingConfig

def load_slides(config: ProcessingConfig) -> Tuple[List[str], Dict[str, str], pd.DataFrame]:
    """Load slide information and initialize processing DataFrame.

    Args:
        config: Configuration object with paths to slide and UUID files.

    Returns:
        Tuple containing:
        - List of valid slide filenames.
        - Dictionary mapping slide filenames to UUIDs.
        - DataFrame for processing control.
    """
    all_data = np.array(pd.read_excel(config.uuid_name_file, engine='openpyxl', header=None))
    slides = []
    id_names = {}
    for data in all_data:
        slides.append(data[1])
        id_names[data[1]] = data[0]

    slides = [slide for slide in slides if os.path.isfile(os.path.join(config.source, id_names[str(slide)], slide))]
    
    if config.process_list:
        df = pd.read_csv(config.process_list)
    else:
        df = initialize_df(slides, config.seg_params, config.filter_params, config.vis_params, config.patch_params)
    
    return slides, id_names, df
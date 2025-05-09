"""
Entry point script for TCGA WSI processing pipeline.

This script loads a YAML configuration file and runs the WSI processing pipeline,
including segmentation, patching, and stitching, for TCGA datasets (e.g., KICH).
"""

import os
import sys
import argparse
import logging

# Set project root for importing custom modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
print(f"Project root set to: {PROJECT_ROOT}")

from src.datasets.preprocessing.config import ProcessingConfig, load_config
from src.datasets.preprocessing.main import main

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

if __name__ == '__main__':
    config_path = parse_args()
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    logger.info(f"Configuration: {config}")
    main(config)
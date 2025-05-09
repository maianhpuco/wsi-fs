"""
Script to generate metadata files (slides.xlsx and uuids.xlsx) for TCGA-KIRP WSI processing.

This script creates two Excel files required by the WSI processing pipeline:
- slides.xlsx: Lists WSI filenames.
- uuids.xlsx: Maps UUIDs to WSI filenames.

It can use a GDC manifest file or scan the WSI directory to infer metadata.
"""

import os
import pandas as pd
import argparse
from typing import List, Tuple

def read_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    """Read UUIDs and filenames from a GDC manifest file.

    Args:
        manifest_path: Path to the GDC manifest file (e.g., gdc_manifest_20180801_125430.txt).

    Returns:
        List of tuples containing (UUID, filename) pairs.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path, sep='\t')
    # GDC manifest typically has columns: id (UUID), filename, md5, size, state
    return list(zip(df['id'], df['filename']))

def scan_directory(source_dir: str) -> List[Tuple[str, str]]:
    """Scan the WSI directory to infer UUIDs and filenames.

    Args:
        source_dir: Path to the directory containing WSI files (e.g., data/tcga_kirp/raw).

    Returns:
        List of tuples containing (UUID, filename) pairs.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    uuid_filename_pairs = []
    for uuid in os.listdir(source_dir):
        uuid_path = os.path.join(source_dir, uuid)
        if os.path.isdir(uuid_path):
            for filename in os.listdir(uuid_path):
                if filename.endswith('.svs'):  # Adjust for other WSI extensions if needed
                    uuid_filename_pairs.append((uuid, filename))
    return uuid_filename_pairs

def generate_metadata(
    source_dir: str,
    metadata_dir: str,
    manifest_path: str = None
) -> None:
    """Generate slides.xlsx and uuids.xlsx files.

    Args:
        source_dir: Path to the directory containing WSI files.
        metadata_dir: Path to save the metadata files.
        manifest_path: Optional path to the GDC manifest file.

    Raises:
        FileNotFoundError: If source_dir or manifest_path is invalid.
        ValueError: If no valid WSIs are found.
    """
    # Get UUID-filename pairs
    if manifest_path:
        uuid_filename_pairs = read_manifest(manifest_path)
    else:
        uuid_filename_pairs = scan_directory(source_dir)

    # Filter valid pairs (check if files exist)
    valid_pairs = []
    for uuid, filename in uuid_filename_pairs:
        file_path = os.path.join(source_dir, uuid, filename)
        if os.path.isfile(file_path):
            valid_pairs.append((uuid, filename))

    if not valid_pairs:
        raise ValueError(f"No valid WSI files found in {source_dir}")

    # Create slides.xlsx
    slides_df = pd.DataFrame([pair[1] for pair in valid_pairs], columns=['Filename'])
    os.makedirs(metadata_dir, exist_ok=True)
    slides_path = os.path.join(metadata_dir, 'slides.xlsx')
    slides_df.to_excel(slides_path, index=False)
    print(f"Generated {slides_path} with {len(slides_df)} slides")

    # Create uuids.xlsx
    uuids_df = pd.DataFrame(valid_pairs, columns=['UUID', 'Filename'])
    uuids_path = os.path.join(metadata_dir, 'uuids.xlsx')
    uuids_df.to_excel(uuids_path, index=False)
    print(f"Generated {uuids_path} with {len(uuids_df)} UUID-filename mappings")

def main():
    """Parse command-line arguments and generate metadata files."""
    parser = argparse.ArgumentParser(description='Generate slides.xlsx and uuids.xlsx for WSI processing')
    parser.add_argument('--config', type=str, help='Path to YAML config file (e.g., configs/data_tgca_kich.yaml)')
    parser.add_argument('--source_dir', type=str, help='Path to WSI files')
    parser.add_argument('--metadata_dir', type=str, help='Path to save metadata files')
    parser.add_argument('--manifest_path', type=str, default=None, help='Path to GDC manifest file')

    args = parser.parse_args()

    # Load config if specified
    if args.config:
        import yaml 
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)   
        source_dir = config.get('source_dir')
        metadata_dir = config.get('metadata_dir')
        manifest_path = config.get('manifest_path', None)
    else:
        source_dir = args.source_dir
        metadata_dir = args.metadata_dir
        manifest_path = args.manifest_path

    generate_metadata(source_dir, metadata_dir, manifest_path)

if __name__ == '__main__':
    main()
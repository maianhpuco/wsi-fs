import os 
import sys 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) 
sys.path.append(PROJECT_ROOT) 

from src.datasets.wsi_core.WholeSlideImage import WholeSlideImage
from src.datasets.wsi_core.wsi_utils import StitchCoords
from src.datasets.wsi_core.batch_process_utils import initialize_df
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import time
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
from src.dataset.wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df



'''
Segmentation of tissue in WSIs

Patching: Extracting image tiles (e.g., 256x256)
Stitching: Reconstructing a heatmap from patches
Configuration handling via YAML or command-line
Batch processing with CSV-based control 
'''


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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

def load_slides(config: ProcessingConfig) -> Tuple[List[str], Dict[str, str], pd.DataFrame]:
    """Load slide information and initialize processing DataFrame."""
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

def update_params_for_slide(df: pd.DataFrame, idx: int, wsi: WholeSlideImage, config: ProcessingConfig, legacy_support: bool) -> Tuple[Dict, Dict, Dict, Dict]:
    """Update parameters for a specific slide, handling legacy support."""
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
    """Segment tissue in the WSI."""
    start_time = time.time()
    wsi.segmentTissue(**seg_params, filter_params=filter_params)
    return wsi, time.time() - start_time

def patch_wsi(wsi: WholeSlideImage, patch_params: Dict, patch_size: int, step_size: int, patch_level: int, save_path: str) -> Tuple[str, float]:
    """Extract patches from the WSI."""
    start_time = time.time()
    patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'save_path': save_path})
    magnification = wsi.wsi.properties['aperio.AppMag']
    patch_params['mag'] = str(magnification)
    file_path = wsi.process_contours(**patch_params)
    return file_path, time.time() - start_time

def stitch_wsi(file_path: str, wsi: WholeSlideImage, downscale: int = 64) -> Tuple[any, float]:
    """Stitch patch coordinates into a heatmap."""
    start_time = time.time()
    heatmap = StitchCoords(file_path, wsi, downscale=downscale, bg_color=(255, 255, 255), alpha=-1, draw_grid=True)
    return heatmap, time.time() - start_time

def process_slide(idx: int, slide: str, df: pd.DataFrame, id_names: Dict[str, str], config: ProcessingConfig, legacy_support: bool) -> Tuple[float, float, float]:
    """Process a single slide (segmentation, patching, stitching)."""
    slide_id, _ = os.path.splitext(slide)
    df.loc[idx, 'process'] = 0

    if config.auto_skip and os.path.isfile(os.path.join(config.patch_save_dir, f"{slide_id}.h5")):
        logger.info(f"{slide_id} already exists, skipping")
        df.loc[idx, 'status'] = 'already_exist'
        return 0.0, 0.0, 0.0

    # Initialize WSI
    full_path = os.path.join(config.source, id_names[slide], slide)
    wsi = WholeSlideImage(full_path)

    # Update parameters
    seg_params, filter_params, vis_params, patch_params = update_params_for_slide(df, idx, wsi, config, legacy_support)

    # Check image size for segmentation
    w, h = wsi.level_dim[seg_params['seg_level']]
    if w * h > 1e8:
        logger.error(f"Image too large for segmentation ({w}x{h}), aborting")
        df.loc[idx, 'status'] = 'failed_seg'
        return 0.0, 0.0, 0.0

    df.loc[idx, 'vis_level'] = vis_params['vis_level']
    df.loc[idx, 'seg_level'] = seg_params['seg_level']

    # Process slide
    seg_time, patch_time, stitch_time = 0.0, 0.0, 0.0
    if config.seg:
        wsi, seg_time = segment_wsi(wsi, seg_params, filter_params)

    if config.save_mask:
        mask, only_mask = wsi.visWSI(**vis_params)
        mask.save(os.path.join(config.mask_save_dir, f"{slide_id}.jpg"))
        only_mask.save(os.path.join(config.only_mask_save_dir, f"{slide_id}.png"))

    if config.patch:
        file_path, patch_time = patch_wsi(wsi, patch_params, config.patch_size, config.step_size, config.patch_level, config.patch_save_dir)

    if config.stitch and os.path.isfile(os.path.join(config.patch_save_dir, f"{slide_id}.h5")):
        heatmap, stitch_time = stitch_wsi(os.path.join(config.patch_save_dir, f"{slide_id}.h5"), wsi)
        heatmap.save(os.path.join(config.stitch_save_dir, f"{slide_id}.jpg"))

    logger.info(f"Segmentation: {seg_time:.2f}s, Patching: {patch_time:.2f}s, Stitching: {stitch_time:.2f}s")
    df.loc[idx, 'status'] = 'processed'
    return seg_time, patch_time, stitch_time

def main(config: ProcessingConfig) -> Tuple[float, float]:
    """Main function to process WSIs."""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.patch_save_dir, exist_ok=True)
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

def parse_args() -> ProcessingConfig:
    """Parse command-line arguments."""
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

if __name__ == '__main__':
    config = parse_args()
    logger.info(f"Configuration: {config}")
    main(config)
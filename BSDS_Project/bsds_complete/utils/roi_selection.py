"""
ROI Selection Utilities for Event Segmentation Analysis
=========================================================

Provides region-specific ROI extraction for event boundary research.

Key regions for event segmentation (Baldassano et al., 2017; Zacks et al., 2007):
- V1: Primary visual cortex
- A1: Primary auditory cortex
- STS: Superior Temporal Sulcus (social/biological motion)
- AG: Angular Gyrus (semantic integration)
- PMC: Posterior Medial Cortex including Precuneus (episodic memory)
- RSC: Retrosplenial Cortex (spatial context)
- mPFC: medial Prefrontal Cortex (schema, situation models)
- Hippocampus: Event encoding and retrieval
- Anterior Insula: Salience, interoception
- Amygdala: Emotional salience

Author: Kyungjin Oh
Date: 2025-12-16
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


# =============================================================================
# Schaefer 400 (7 Networks) ROI Mapping to Event Segmentation Regions
# =============================================================================

# These mappings are based on Schaefer et al. (2018) parcellation
# ROI indices are 1-based (matching atlas labels), converted to 0-based in code

SCHAEFER400_EVENT_ROIS = {
    # Primary Visual Cortex (V1) - Visual network, posterior occipital
    'V1': {
        'description': 'Primary Visual Cortex',
        'network': 'Visual',
        'schaefer400_labels': [
            # Left hemisphere V1
            '7Networks_LH_Vis_1', '7Networks_LH_Vis_2', '7Networks_LH_Vis_3',
            '7Networks_LH_Vis_4', '7Networks_LH_Vis_5',
            # Right hemisphere V1
            '7Networks_RH_Vis_1', '7Networks_RH_Vis_2', '7Networks_RH_Vis_3',
            '7Networks_RH_Vis_4', '7Networks_RH_Vis_5',
        ],
        # Approximate indices in Schaefer 400 (0-based)
        'indices_400': list(range(0, 5)) + list(range(200, 205)),  # ~10 ROIs
    },

    # Primary Auditory Cortex (A1) - SomMot network, superior temporal
    'A1': {
        'description': 'Primary Auditory Cortex',
        'network': 'SomMot',
        'schaefer400_labels': [
            '7Networks_LH_SomMot_1', '7Networks_LH_SomMot_2',
            '7Networks_RH_SomMot_1', '7Networks_RH_SomMot_2',
        ],
        # Auditory cortex parcels in Schaefer (temporal part of SomMot)
        'indices_400': [31, 32, 33] + [231, 232, 233],  # ~6 ROIs
    },

    # Superior Temporal Sulcus (STS) - Social perception, voice, biological motion
    'STS': {
        'description': 'Superior Temporal Sulcus',
        'network': 'SalVentAttn/Default',
        'schaefer400_labels': [
            '7Networks_LH_SalVentAttn_TempOccPar_1',
            '7Networks_LH_Default_Temp_1', '7Networks_LH_Default_Temp_2',
            '7Networks_RH_SalVentAttn_TempOccPar_1',
            '7Networks_RH_Default_Temp_1', '7Networks_RH_Default_Temp_2',
        ],
        'indices_400': [91, 92, 176, 177] + [291, 292, 376, 377],  # ~8 ROIs
    },

    # Angular Gyrus (AG) - Semantic integration, episodic retrieval
    'AG': {
        'description': 'Angular Gyrus',
        'network': 'Default',
        'schaefer400_labels': [
            '7Networks_LH_Default_Par_1', '7Networks_LH_Default_Par_2',
            '7Networks_RH_Default_Par_1', '7Networks_RH_Default_Par_2',
        ],
        'indices_400': [182, 183, 184] + [382, 383, 384],  # ~6 ROIs
    },

    # Posterior Medial Cortex (PMC) including Precuneus
    'PMC': {
        'description': 'Posterior Medial Cortex (Precuneus)',
        'network': 'Default',
        'schaefer400_labels': [
            '7Networks_LH_Default_PCC_1', '7Networks_LH_Default_PCC_2',
            '7Networks_LH_Default_PCC_3',
            '7Networks_RH_Default_PCC_1', '7Networks_RH_Default_PCC_2',
            '7Networks_RH_Default_PCC_3',
        ],
        'indices_400': [188, 189, 190, 191] + [388, 389, 390, 391],  # ~8 ROIs
    },

    # Retrosplenial Cortex (RSC) - Spatial context, scene processing
    'RSC': {
        'description': 'Retrosplenial Cortex',
        'network': 'Default',
        'schaefer400_labels': [
            '7Networks_LH_Default_PCC_4', '7Networks_LH_Default_PCC_5',
            '7Networks_RH_Default_PCC_4', '7Networks_RH_Default_PCC_5',
        ],
        'indices_400': [192, 193] + [392, 393],  # ~4 ROIs
    },

    # medial Prefrontal Cortex (mPFC) - Schema, situation models
    'mPFC': {
        'description': 'medial Prefrontal Cortex',
        'network': 'Default',
        'schaefer400_labels': [
            '7Networks_LH_Default_PFC_1', '7Networks_LH_Default_PFC_2',
            '7Networks_LH_Default_PFC_3', '7Networks_LH_Default_PFC_4',
            '7Networks_RH_Default_PFC_1', '7Networks_RH_Default_PFC_2',
            '7Networks_RH_Default_PFC_3', '7Networks_RH_Default_PFC_4',
        ],
        'indices_400': [194, 195, 196, 197, 198, 199] + [394, 395, 396, 397, 398, 399],  # ~12 ROIs
    },

    # Anterior Insula - Salience detection
    'AntInsula': {
        'description': 'Anterior Insula',
        'network': 'SalVentAttn',
        'schaefer400_labels': [
            '7Networks_LH_SalVentAttn_FrOperIns_1',
            '7Networks_LH_SalVentAttn_FrOperIns_2',
            '7Networks_RH_SalVentAttn_FrOperIns_1',
            '7Networks_RH_SalVentAttn_FrOperIns_2',
        ],
        'indices_400': [100, 101, 102] + [300, 301, 302],  # ~6 ROIs
    },
}

# Subcortical regions need Harvard-Oxford or Freesurfer atlas
SUBCORTICAL_EVENT_ROIS = {
    'Hippocampus': {
        'description': 'Hippocampus (bilateral)',
        'harvard_oxford_labels': ['Left Hippocampus', 'Right Hippocampus'],
        'freesurfer_labels': ['Left-Hippocampus', 'Right-Hippocampus'],
    },
    'Amygdala': {
        'description': 'Amygdala (bilateral)',
        'harvard_oxford_labels': ['Left Amygdala', 'Right Amygdala'],
        'freesurfer_labels': ['Left-Amygdala', 'Right-Amygdala'],
    },
}


def get_event_segmentation_rois(
    n_rois: int = 400,
    include_subcortical: bool = True,
    regions: Optional[List[str]] = None
) -> Dict:
    """
    Get ROI indices for event segmentation analysis.

    Args:
        n_rois: Number of ROIs in Schaefer atlas (100, 200, 400)
        include_subcortical: Whether to include hippocampus/amygdala
        regions: Specific regions to include. If None, includes all.
                 Options: 'V1', 'A1', 'STS', 'AG', 'PMC', 'RSC', 'mPFC',
                          'AntInsula', 'Hippocampus', 'Amygdala'

    Returns:
        Dictionary with region info and indices
    """
    if regions is None:
        regions = list(SCHAEFER400_EVENT_ROIS.keys())
        if include_subcortical:
            regions += list(SUBCORTICAL_EVENT_ROIS.keys())

    result = {
        'cortical_indices': [],
        'cortical_regions': {},
        'subcortical_regions': [],
        'n_rois': n_rois,
        'total_rois': 0,
    }

    # Scale factor for different atlas resolutions
    scale = n_rois / 400

    for region in regions:
        if region in SCHAEFER400_EVENT_ROIS:
            roi_info = SCHAEFER400_EVENT_ROIS[region]

            # Scale indices for atlas resolution
            if n_rois == 400:
                indices = roi_info['indices_400']
            elif n_rois == 200:
                # Approximate mapping for 200 ROIs
                indices = [int(i * 0.5) for i in roi_info['indices_400']]
                indices = list(set(indices))  # Remove duplicates
            elif n_rois == 100:
                indices = [int(i * 0.25) for i in roi_info['indices_400']]
                indices = list(set(indices))
            else:
                indices = [int(i * scale) for i in roi_info['indices_400']]
                indices = list(set(indices))

            # Ensure indices are valid
            indices = [i for i in indices if 0 <= i < n_rois]

            result['cortical_indices'].extend(indices)
            result['cortical_regions'][region] = {
                'description': roi_info['description'],
                'network': roi_info['network'],
                'indices': indices,
                'n_rois': len(indices),
            }

        elif region in SUBCORTICAL_EVENT_ROIS and include_subcortical:
            result['subcortical_regions'].append(region)

    # Remove duplicates and sort
    result['cortical_indices'] = sorted(list(set(result['cortical_indices'])))
    result['total_rois'] = len(result['cortical_indices']) + len(result['subcortical_regions']) * 2

    return result


def create_event_roi_masker(
    n_rois: int = 400,
    include_subcortical: bool = True,
    regions: Optional[List[str]] = None,
    standardize: bool = True,
    memory: str = 'nilearn_cache'
):
    """
    Create a NiftiMasker for event segmentation ROIs.

    Args:
        n_rois: Number of ROIs in Schaefer atlas
        include_subcortical: Whether to include hippocampus/amygdala
        regions: Specific regions to include
        standardize: Whether to z-score the time series
        memory: Nilearn cache directory

    Returns:
        Tuple of (masker, roi_info)
    """
    try:
        from nilearn import datasets, image
        from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
        import nibabel as nib
    except ImportError:
        raise ImportError("nilearn required. Install with: pip install nilearn")

    roi_info = get_event_segmentation_rois(n_rois, include_subcortical, regions)

    # Load Schaefer atlas
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7)
    schaefer_img = nib.load(schaefer.maps)
    schaefer_data = schaefer_img.get_fdata()

    # Create mask with only selected cortical ROIs
    selected_mask = np.zeros_like(schaefer_data)
    new_label = 1
    label_mapping = {}

    for idx in roi_info['cortical_indices']:
        atlas_label = idx + 1  # Schaefer uses 1-based labels
        selected_mask[schaefer_data == atlas_label] = new_label
        label_mapping[new_label] = {
            'original_idx': idx,
            'original_label': atlas_label,
        }
        new_label += 1

    # Add subcortical if requested
    if include_subcortical and roi_info['subcortical_regions']:
        try:
            # Load Harvard-Oxford subcortical atlas
            ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            ho_img = nib.load(ho_sub.maps)
            ho_data = ho_img.get_fdata()
            ho_labels = ho_sub.labels

            # Resample to Schaefer space if needed
            if ho_data.shape != schaefer_data.shape:
                ho_img_resampled = image.resample_to_img(ho_img, schaefer_img, interpolation='nearest')
                ho_data = ho_img_resampled.get_fdata()

            # Add hippocampus
            if 'Hippocampus' in roi_info['subcortical_regions']:
                for i, label in enumerate(ho_labels):
                    if 'Hippocampus' in label:
                        selected_mask[ho_data == i] = new_label
                        label_mapping[new_label] = {'region': label}
                        new_label += 1

            # Add amygdala
            if 'Amygdala' in roi_info['subcortical_regions']:
                for i, label in enumerate(ho_labels):
                    if 'Amygdala' in label:
                        selected_mask[ho_data == i] = new_label
                        label_mapping[new_label] = {'region': label}
                        new_label += 1

        except Exception as e:
            warnings.warn(f"Could not load subcortical atlas: {e}. Using cortical only.")

    # Create new atlas image
    selected_img = nib.Nifti1Image(selected_mask, schaefer_img.affine)

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=selected_img,
        standardize=standardize,
        memory=memory,
        verbose=0
    )

    roi_info['label_mapping'] = label_mapping
    roi_info['n_selected'] = new_label - 1

    return masker, roi_info


def extract_event_roi_timeseries(
    func_path: str,
    n_rois: int = 400,
    include_subcortical: bool = True,
    regions: Optional[List[str]] = None,
    standardize: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Extract time series from event segmentation ROIs.

    Args:
        func_path: Path to functional NIfTI file
        n_rois: Number of ROIs in Schaefer atlas
        include_subcortical: Whether to include hippocampus/amygdala
        regions: Specific regions to include
        standardize: Whether to z-score

    Returns:
        Tuple of (time_series array [T x n_selected_rois], roi_info dict)
    """
    masker, roi_info = create_event_roi_masker(
        n_rois=n_rois,
        include_subcortical=include_subcortical,
        regions=regions,
        standardize=standardize
    )

    ts = masker.fit_transform(func_path)
    roi_info['shape'] = ts.shape

    return ts, roi_info


def get_region_summary() -> str:
    """Get a summary of available event segmentation regions."""
    lines = [
        "Event Segmentation ROIs:",
        "=" * 50,
        "\nCortical Regions (Schaefer 400):",
    ]

    for name, info in SCHAEFER400_EVENT_ROIS.items():
        n_rois = len(info['indices_400'])
        lines.append(f"  {name:12s}: {info['description']:35s} ({n_rois} ROIs, {info['network']})")

    lines.append("\nSubcortical Regions (Harvard-Oxford):")
    for name, info in SUBCORTICAL_EVENT_ROIS.items():
        lines.append(f"  {name:12s}: {info['description']}")

    lines.append("\n" + "=" * 50)
    total_cortical = sum(len(info['indices_400']) for info in SCHAEFER400_EVENT_ROIS.values())
    lines.append(f"Total: ~{total_cortical} cortical + 4 subcortical ROIs")

    return "\n".join(lines)


# Quick reference for command line
EVENT_ROI_PRESETS = {
    'full': {
        'regions': None,  # All regions
        'include_subcortical': True,
        'description': 'All event segmentation ROIs (~64 ROIs)',
    },
    'cortical': {
        'regions': ['V1', 'A1', 'STS', 'AG', 'PMC', 'RSC', 'mPFC', 'AntInsula'],
        'include_subcortical': False,
        'description': 'Cortical event ROIs only (~60 ROIs)',
    },
    'dmn': {
        'regions': ['AG', 'PMC', 'RSC', 'mPFC'],
        'include_subcortical': False,
        'description': 'Default Mode Network event ROIs (~30 ROIs)',
    },
    'memory': {
        'regions': ['PMC', 'RSC', 'mPFC', 'Hippocampus'],
        'include_subcortical': True,
        'description': 'Memory-related ROIs (~28 ROIs)',
    },
    'sensory': {
        'regions': ['V1', 'A1', 'STS'],
        'include_subcortical': False,
        'description': 'Sensory processing ROIs (~24 ROIs)',
    },
}


if __name__ == '__main__':
    print(get_region_summary())
    print("\n\nPresets:")
    for name, preset in EVENT_ROI_PRESETS.items():
        print(f"  {name}: {preset['description']}")

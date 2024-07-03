"""Segmentation ensembles.
"""
from pathlib import Path
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon2mask
from skimage.transform import resize
import matplotlib.pyplot as plt

def get_han_ensemble(data_dir, num_rows=None, num_cols=None, patient_id="HCAI-036", structure_name="Parotid_R", slice_num=41, iso_value=0.8):
    
    data_dir = Path(data_dir)
    assert data_dir.exists()

    fn = data_dir.joinpath(f"ensemble-{structure_name}-hptc/{patient_id}/ed_ensemble-v4_size-subsets-31-{structure_name}.npz")
    assert fn.exists()

    whole_vol = slice_num is None
    return_masks = iso_value is not None
    dataset = np.load(fn)

    img = dataset["img"]
    gt = dataset["gt"]
    ensemble_probs = [v for k, v in dataset.items() if "ep" in k]

    num_slices = img.shape[0]
    if num_rows is None:
        num_rows = img.shape[1]
    if num_cols is None:
        num_cols = img.shape[2]

    if not whole_vol:
        img = img[slice_num]
        gt = gt[slice_num]
        ensemble_probs = [ep[slice_num] for ep in ensemble_probs]    

    shape_ensemble = ensemble_probs[0].shape

    if return_masks:
        ensemble_masks = [(member > 0.8).astype(float) for member in ensemble_probs]
    else:
        ensemble_masks = ensemble_probs
    
    if whole_vol:         
        target_size = (num_slices, num_rows, num_cols)
    else:
        target_size = (num_rows, num_cols)

    img = resize(img, target_size, order=1)
    gt = resize(gt, target_size, order=0)
    ensemble_masks = [resize(mask, target_size, order=0) for mask in ensemble_masks]

    return img, gt, ensemble_masks
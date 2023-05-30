import numpy as np

def _is_valid_mask(mask):
    values = np.unique(mask)
    if len(values) != 2:
        return False
    if values[0] != 0 or values[1] != 255:
        return False
    return True

def judge(mask, size_ratio=[1, 4], pixel_ratio=[0.01, 1]):
    if _is_valid_mask(mask):
        mask_size_ratio = max(mask.shape) / min(mask.shape)
        if mask_size_ratio < size_ratio[0] or mask_size_ratio > size_ratio[1]:
            return False

        bincounts = np.bincount(mask.ravel())
        mask_pixel_ratio = bincounts[0] / bincounts[255]
        if mask_pixel_ratio < pixel_ratio[0] or mask_pixel_ratio > pixel_ratio[1]:
            return False
        return True
    else:
        return False
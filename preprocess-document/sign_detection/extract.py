from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np

def extract(mask, outlier_weight=3, outlier_bias=100, amplfier=10, min_area_size=10):
    """
    params
    ------
    mask: numpy array
        The mask of the image. It's calculated by Loader.
    return
    ------
    labeled_image: numpy array
        The labeled image.
        The numbers in the array are the region labels.
    """
    condition = mask > mask.mean()
    labels = measure.label(condition, background=1)

    total_pixels = 0
    nb_region = 0
    average = 0.0
    for region in regionprops(labels):
        if region.area > min_area_size:
            total_pixels += region.area
            nb_region += 1
    
    if nb_region > 1:
        average = total_pixels / nb_region
        # small_size_outlier is used as a threshold value to remove pixels
        # are smaller than small_size_outlier
        small_size_outlier = average * outlier_weight + outlier_bias

        # big_size_outlier is used as a threshold value to remove pixels
        # are bigger than big_size_outlier
        big_size_outlier = small_size_outlier * amplfier

        # remove small pixels
        labeled_image = morphology.remove_small_objects(labels, small_size_outlier)
        # remove the big pixels
        component_sizes = np.bincount(labeled_image.ravel())
        too_small = component_sizes > (big_size_outlier)
        too_small_mask = too_small[labeled_image]
        labeled_image[too_small_mask] = 0

        labeled_mask = np.full(labeled_image.shape, 255, dtype="uint8")
        labeled_mask = labeled_mask * (labeled_image == 0)
    else:
        labeled_mask = mask

    return labeled_mask
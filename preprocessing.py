import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import label
from scipy import ndimage as ndi

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025



def load_nifti(nifti_path: str) -> np.ndarray:
    """
    Load a NIfTI file and return the data as a NumPy array.
    """
    print(f'Running load_nifti')
    img = nib.load(nifti_path)
    data = img.get_fdata()

    return data


def threshold_volume(data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Apply Otsu's thresholding to a 3D volume.
    """
    print(f'Running threshold_volume')
    data_flat = data.ravel()  # A view for C-contiguous inputs; same voxel order.
    threshold_value = threshold_otsu(data_flat)
    binary_volume = data > threshold_value # Apply thresholding

    return binary_volume, threshold_value


# def extract_largest_blob(volume: np.ndarray) -> np.ndarray:
#    """
#    Extract the largest blob from a binary volume.
#    """
#    print(f'Running extract_largest_blob')

#    labeled_volume, num_features = label(volume)

#    if num_features == 0:
#        raise ValueError("No blobs found in the volume.")

#    blob_sizes = np.bincount(labeled_volume.ravel()) # Count the number of voxels in each blob
#    blob_sizes[0] = 0 # Background is not a blob, set its size to 0
#    largest_blob_label = blob_sizes.argmax() # Find the label of the largest blob

#    largest_blob_mask = (labeled_volume == largest_blob_label)

#    return largest_blob_mask

def extract_largest_blob(volume: np.ndarray) -> np.ndarray:
    """
    Extract the largest blob from a binary volume.
    """
    print(f'Running extract_largest_blob')

    labeled_volume, num_features = label(volume)

    if num_features == 0:
        raise ValueError("No blobs found in the volume.")

    blob_sizes = np.bincount(labeled_volume.ravel()) # Count the number of voxels in each blob
    blob_sizes[0] = 0 # Background is not a blob, set its size to 0
    largest_blob_label = blob_sizes.argmax() # Find the label of the largest blob

    largest_blob_mask = (labeled_volume == largest_blob_label)

    return largest_blob_mask

# def extract_largest_blob(volume: np.ndarray, connectivity: int = 1) -> np.ndarray:
#     """
#     Return a Boolean mask that selects the largest connected component
#     (―“blob”―) in a binary 2-D or 3-D volume.

#     Parameters
#     ----------
#     volume : ndarray
#         Binary input. Non-zero voxels are considered foreground.
#     connectivity : {1, 2, 3}, optional
#     Neighbourhood definition  
#     (in 3-D: 1 → 6-connected, 2 → 18-connected, 3 → 26-connected).

#     Returns
#     -------
#     mask : ndarray of bool, same shape as *volume*
#         True where the largest blob is present, False elsewhere.
#     """

#     print(f'Running extract_largest_blob')
# #     ── 1. Ensure a compact dtype; this also coerces any non-binary input  ──
#     volume = np.asarray(volume, dtype=bool)

# #     ── 2. Connected-component labelling (executed in C) ────────────────
#     structure = ndi.generate_binary_structure(volume.ndim, connectivity)
#     labeled, num = ndi.label(volume, structure=structure)

#     if num == 0:                       # no foreground at all
#         return np.zeros_like(volume, dtype=bool)
#     if num == 1:                       # already a single blob
#         return volume

# #     ── 3. Find the component with the most voxels (O(N) time, O(K) mem) ─
#     counts = np.bincount(labeled.ravel())
#     largest_label = counts[1:].argmax() + 1   # skip background bin 0

# #     ── 4. Boolean mask – broadcasting is done in C, so this is fast ─────
#     return labeled == largest_label


def preprocessing(nifti_path: str) -> tuple[np.ndarray, float]:
    """
    Preprocess a NIfTI file by applying thresholding and extracting the largest blob.
    """
    print(f'Running preprocessing')

    data = load_nifti(nifti_path)
    binary_volume, threshold_value = threshold_volume(data)
    del data  # Labeling needs only the binary volume.
    largest_blob_mask = extract_largest_blob(binary_volume)

    return largest_blob_mask, threshold_value
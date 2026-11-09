from preprocessing import *
from PCA_resample import *
from find_center_point import *
from graph_cut_funcs import *

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025


########### FILL OUT ###########
path = "C:/Users/chris/Mit drev/DTU/4. Endocasting/Data/human_segmented_masked.nii.gz"  # Replace with your actual NifTi file path
out_path = "C:/Users/chris/Mit drev/DTU/4. Endocasting/Data/Subset_Skull_Volumes/processed/" # Replace with your desired output path
num_iters = [10,5,3,2]
step_sizes = [1,1,0.5,0.5]
use_pca_resampling = True  # False: automatic center selection in the original volume.
remesh = True  # Once per stage: after iterations 5, 3, 2, 1 for the setup above.
################################

if remesh:
    import pymeshlab  # Check the added dependency before processing the scan.



# Use the same volume for center detection, mesh sampling and mask export.
volume_path = path
if use_pca_resampling:
    _ = PCA_resample_GUI(path)
    volume_path = path.replace('.nii', '_PCA_resampled.nii')

local_max, local_max_values, points = find_center_points(volume_path)
print(f'Number of local maxima: {len(local_max)}')


if use_pca_resampling:
    center_point, min_dist = select_local_max(points, local_max, local_max_values)
else:
    # The bottleneck/enclosure method returns the best candidate first.
    center_point, min_dist = local_max[0], local_max_values[0]
print(f'Selected center point: {center_point}')
print(f'Minimum distance value at selected center point: {min_dist}')

#iterative_mesh(center_point, min_dist, out_path, volume_path) #outputs the first iterative mesh and last
iterative_mesh_every_iter(center_point, min_dist, out_path, volume_path,num_iters=num_iters,step_sizes=step_sizes,remesh=remesh) #outputs all iterative meshes

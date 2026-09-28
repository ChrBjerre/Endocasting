from preprocessing import preprocessing
from cavity_ranking import rank_cavity_candidates
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import nibabel as nib
import os

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025





def get_alpha_shape(nifti_path_pca, subsampling_factor = None, alpha_value = None):

    import pyvista as pv

    print(f'Running get_alpha_shape')

    # Load the NIfTI file and preprocess it
    largest_blob_mask, _ = preprocessing(nifti_path_pca)

    # Find the indices of the voxels that are inside the mask
    points = np.argwhere(largest_blob_mask)

    # Check, if the user provided a subsampling factor and alpha value
    # Otherwise use default values
    if subsampling_factor is None:
        subsampling_factor = int(np.ceil(len(points)/5000))
    if alpha_value is None:
        alpha_value = 30

    points_new = points[::subsampling_factor]

    points_new = np.array(points_new, dtype=np.float32)

    cloud = pv.PolyData(points_new)
    hull = cloud.delaunay_3d(alpha=0)  # alpha=0 forces convex hull
    shell = hull.extract_geometry()

    shell.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=True)
    shell.save('./temp_object.obj')

    return points_new, largest_blob_mask


def get_voxels_in_mesh(nifti_path_pca, obj_path="./temp_object.obj"):
    import vtk
    from vtk.util import numpy_support

    nifti_img = nib.load(nifti_path_pca)
    nifti_affine = nifti_img.affine
    nifti_shape = nifti_img.shape
    spacing = nifti_affine[:3, :3].diagonal()
    origin = nifti_affine[:3, 3]

    # --- load OBJ (already in voxel space) -------------------------------
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_path)
    reader.Update()
    mesh = reader.GetOutput()


    # === Apply affine transform to bring mesh into world (image) space ===
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, nifti_affine[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(mesh)
    transform_filter.Update()

    mesh_transformed = transform_filter.GetOutput()

    # === Prepare image volume ===
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nifti_shape)
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(origin)

    # === Convert mesh to image stencil ===
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(mesh_transformed)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputSpacing(spacing)
    poly_to_stencil.SetOutputWholeExtent(image_data.GetExtent())
    poly_to_stencil.Update()

    # === Convert stencil to binary image ===
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(poly_to_stencil.GetOutputPort())
    stencil_to_image.SetInsideValue(1)
    stencil_to_image.SetOutsideValue(0)
    stencil_to_image.Update()

    # === Convert to NumPy array ===
    vtk_image = stencil_to_image.GetOutput()
    dims = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()

    np_array = numpy_support.vtk_to_numpy(scalars)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])
    np_array = np.transpose(np_array, (2, 1, 0))  # Match NIfTI axes

    mesh_mask = np_array

    return mesh_mask





def _legacy_candidate_points(largest_blob_mask, mesh_mask, sigma=3):

    print(f'Running get_candidate_points')

    # Compute the Euclidean distance map (on inverted, so that the distance is computed from cavity to skull).
    distance_map = distance_transform_edt(largest_blob_mask == 0)
    #np.save('distance_map.npy', distance_map)

    # Apply Gaussian filter to the distance map
    # Reuse the float64 distance buffer; the unfiltered map is no longer needed.
    gaussian_filter(distance_map, sigma=sigma, output=distance_map)
    distance_map *= mesh_mask
    masked_distance_map = distance_map
    #np.save('masked_distance_map.npy', masked_distance_map)

    local_max = peak_local_max(masked_distance_map, min_distance=50, threshold_abs=0.1, exclude_border=True)

    # Extract values at local maxima
    local_max_values = np.array([masked_distance_map[tuple(coord)] for coord in local_max])

    return local_max, local_max_values


def get_candidate_points(largest_blob_mask, mesh_mask=None, sigma=3, *,
                         method="bottleneck", return_scores=False, **ranking_options):
    """Return ranked original-array indices and EDT radii in voxel units.

    The old hull mask is used only with method="legacy". A hull is not evidence
    of enclosure. With return_scores=True, append a CavityCandidates record.
    """
    if method == "legacy":
        if mesh_mask is None or return_scores or ranking_options:
            raise ValueError("Legacy mode requires mesh_mask and does not provide ranking scores")
        return _legacy_candidate_points(largest_blob_mask, mesh_mask, sigma)
    if method != "bottleneck":
        raise ValueError("method must be 'bottleneck' or 'legacy'")
    result = rank_cavity_candidates(largest_blob_mask, sigma=sigma, **ranking_options)
    values = (result.points, result.peak)
    return (*values, result) if return_scores else values


def find_center_points(nifti_path_pca, subsampling_factor=None, alpha_value=None,
                       sigma=3, *, method="bottleneck", return_scores=False,
                       **ranking_options):
    """Keep the three-value pipeline interface, with the recommended seed first.

    The returned distance is the unsmoothed original EDT at that seed, never
    the ranking score. This keeps downstream sphere radii in voxel units.
    method="legacy" retains the original hull-based behaviour for comparison.
    """
    if method == "legacy":
        if return_scores or ranking_options:
            raise ValueError("Legacy mode does not provide ranking scores/options")
        points, bone = get_alpha_shape(nifti_path_pca, subsampling_factor, alpha_value)
        candidates, values = _legacy_candidate_points(bone, get_voxels_in_mesh(nifti_path_pca), sigma)
        return candidates, values, points
    if method != "bottleneck":
        raise ValueError("method must be 'bottleneck' or 'legacy'")
    bone, _ = preprocessing(nifti_path_pca)
    result = rank_cavity_candidates(bone, sigma=sigma, **ranking_options)
    # A bounded point cloud for the existing GUI only; it is not a hull or mask.
    step = max(1, int(np.ceil(max(bone.shape) / 128)))
    points = np.argwhere(bone[::step, ::step, ::step]) * step
    factor = subsampling_factor if subsampling_factor is not None else max(1, int(np.ceil(len(points) / 5000)))
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("subsampling_factor must be a positive integer")
    points = points[::factor].astype(np.float32)
    values = (result.points, result.peak, points)
    return (*values, result) if return_scores else values



def select_local_max(points, local_max, local_max_values):
    """
    Opens a Tkinter GUI containing a 3D scatter plot:
      - All `points` (Nx3) plotted in black
      - All `local_max` (Mx3) plotted in red
      - One selected local_max at a time is shown bigger in green.
    The user can cycle through local_max points using "Previous" or "Next"
    and choose the final selection with "Done". The chosen point is returned.
    """

    if len(local_max) == 0:
        raise ValueError("No candidate points to select")

    # --- Create the main Tkinter window ---
    root = tk.Tk()
    root.title("Select local_max Point")

    # Keep track of the current index and the final chosen point
    current_index = [0]  # wrap in list to mutate inside nested function
    chosen_point = [None]
    chosen_point_value = [None]

    # --- Create a Matplotlib figure and 3D axis ---
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Plot all the background points in black ---
    if len(points) > 0:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, s=5, c='k', marker='.', alpha=0.8)
    
    # --- Plot all local_max in red (small-ish size) ---
    # We'll keep them in one scatter for simplicity
    local_max_x, local_max_y, local_max_z = local_max[:, 0], local_max[:, 1], local_max[:, 2]
    ax.scatter(local_max_x, local_max_y, local_max_z, s=50, c="red")

    # --- Create a single scatter for the selected local_max point (green) ---
    # Initialize with the first local_max point
    selected_scatter = ax.scatter(
        [local_max[0, 0]],
        [local_max[0, 1]],
        [local_max[0, 2]],
        s=100,  # larger size
        c="green"
    )

    # --- Set up axes labeling/limits (optional) ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect('equal')
    # If you want to set some nice view, you can do so here
    # ax.view_init(elev=20, azim=30)

    # --- Function to update the selected scatter point ---
    def update_selected_point():
        idx = current_index[0]
        selected_scatter._offsets3d = (
            [local_max[idx, 0]],
            [local_max[idx, 1]],
            [local_max[idx, 2]]
        )
        # Force redraw
        canvas.draw()

    # --- Next / Previous button callbacks ---
    def next_point():
        current_index[0] = (current_index[0] + 1) % len(local_max)
        update_selected_point()

    def prev_point():
        current_index[0] = (current_index[0] - 1) % len(local_max)
        update_selected_point()

    # --- Done button callback ---
    def done():
        # Store the chosen point and close the Tkinter window
        chosen_point[0] = local_max[current_index[0], :]
        chosen_point_value[0] = local_max_values[current_index[0]]
        root.quit()  # breaks out of root.mainloop()
        root.destroy()

    # --- Embed the figure in the Tkinter window ---
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Create the frame for the buttons ---
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    btn_prev = tk.Button(button_frame, text="Previous", command=prev_point)
    btn_prev.pack(side=tk.LEFT, padx=5, pady=5)

    btn_next = tk.Button(button_frame, text="Next", command=next_point)
    btn_next.pack(side=tk.LEFT, padx=5, pady=5)

    btn_done = tk.Button(button_frame, text="Done", command=done)
    btn_done.pack(side=tk.RIGHT, padx=5, pady=5)

    # --- Start the Tk event loop ---
    root.mainloop()

    # --- Return the final chosen point after the window closes ---
    return chosen_point[0], chosen_point_value[0]
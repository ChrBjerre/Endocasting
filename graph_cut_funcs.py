import maxflow
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from icosphere import icosphere
from pygel3d import hmesh
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from vtk.util import numpy_support
import trimesh
import open3d as o3d
import vtk

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025




def create_icosphere(center_point, nu=10, radius=10):
    """
    Create an icosphere mesh centered at a specified point.
    Parameters:
    - center_point: An array [x,y,z] representing the center of the icosphere.
    - nu: The level of subdivision for the icosphere. Higher values create more vertices.
    - radius: The radius of the icosphere.
    Returns:
    - A pygel3d hmesh object representing the icosphere mesh.
    """

    # Generate the icosphere
    vertices, faces = icosphere(nu)
    vertices *= radius

    # Center icosphere at the specified point
    vertices = vertices + center_point

    # Ensure vertices and faces are NumPy arrays with correct data types
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    return hmesh.Manifold.from_triangles(vertices, faces)



def create_neighbor_list(mesh):
    """
    Create a list of neighbors for each vertex in the mesh.
    Parameters:
    - mesh: A pygel3d hmesh object representing the mesh.
    Returns:
    - A list of NumPy arrays, where each array contains the indices of neighboring vertices for each vertex in the mesh.
    """

    # Make a list of neighbors for each vertex
    neighbor_list = []

    for i in mesh.vertices():
        neighbors = np.array(mesh.circulate_vertex(i, mode='v'))
        neighbor_list.append(neighbors)
    
    return neighbor_list



def create_sample_grid_vec(mesh, steps, step_size):
    """
    Create a sample grid of points along the normals of the mesh vertices.
    Parameters:
    - mesh: A pygel3d hmesh object representing the mesh.
    - steps: The number of steps to take along the normal direction.
    - step_size: The distance to move along the normal direction for each step.
    Returns:
    - A NumPy array of shape (steps+1, n_vertices, 3) containing the sampled points.
    """

    vertices = list(mesh.vertices())

    # Precompute positions and normalized normals
    positions = np.array([mesh.positions()[v] for v in vertices])
    normals = np.array([mesh.vertex_normal(v) for v in vertices])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize all normals

    # Create step offsets: shape (steps+1, 1, 1)
    step_offsets = np.arange(steps + 1)[:, None, None] * step_size

    # Broadcast: shape (steps+1, n_vertices, 3)
    samples = positions[None, :, :] + step_offsets * normals[None, :, :]

    return samples


def create_sample_grid_vec_new(mesh, steps, step_size, *, frac=1.0):
    """
    Create a sample grid of points along the vertex normals, optionally also
    stepping *into* the surface.

    Parameters
    ----------
    mesh : pygel3d hmesh object
        The mesh whose vertex normals are followed.
    steps : int
        Total number of *non-zero* steps that will be distributed between
        the outward (+) and inward (−) directions.  The returned tensor
        therefore has `steps + 1` slices (the extra slice is the original
        vertex position at offset 0).
    step_size : float
        Distance between two successive samples.
    frac : float, default 1.0
        Fraction of the `steps` that should go outward.
        * `frac = 1.0` reproduces the old behaviour (only + direction).
        * `frac = 0.9` → 90 % outward, 10 % inward.
        * `frac = 0.0` → only inward sampling.

    Returns
    -------
    samples : (steps + 1, n_vertices, 3) ndarray
        Ordered as

        inward … −3·h -2·h -1·h, 0, +1·h +2·h … outward,
        where `h = step_size`.
    """
    # -------------------------------------------------
    #  1. Number of steps in each direction
    # -------------------------------------------------
    n_out = int(round(steps * frac))
    n_in  = steps - n_out                      # always keeps   n_in + n_out == steps

    # -------------------------------------------------
    #  2. Offsets (shape  (steps+1,) )
    #     order:  farthest-in  …  -h, 0, +h  …  farthest-out
    # -------------------------------------------------
    neg_offsets = -np.arange(n_in, 0, -1, dtype=float)   #  -n_in·h  …  -h
    pos_offsets =  np.arange(1, n_out + 1, dtype=float)  #   +h     …  +n_out·h
    offsets     = np.concatenate((neg_offsets, [0.0], pos_offsets))

    # -------------------------------------------------
    #  3. Vertex positions & unit normals  ------------  (same cost as before)
    # -------------------------------------------------
    vertices  = list(mesh.vertices())
    positions = np.asarray([mesh.positions()[v]      for v in vertices])
    normals   = np.asarray([mesh.vertex_normal(v)    for v in vertices])
    normals  /= np.linalg.norm(normals, axis=1, keepdims=True)

    # -------------------------------------------------
    #  4. Broadcast: (steps+1, n_vertices, 3)
    # -------------------------------------------------
    samples = (offsets[:, None, None] * step_size) * normals[None, :, :]
    samples += positions[None, :, :]  # Reuse the offset buffer for final coordinates.

    return samples



def prepare_nifti_interpolator(nifti_path: str) -> RegularGridInterpolator:
    """
    Loads a NIfTI file and returns a RegularGridInterpolator for its data.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])

    interpolator = RegularGridInterpolator(
        (x, y, z),
        data,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    return interpolator



def sample_in_nifti(samples, interpolator, show_plot: bool = False) -> np.ndarray:
    """
    Sample the values at the given coordinates in a NIfTI file.
    Parameters:
    - samples: A NumPy array of shape (N, M, 3) containing the coordinates to sample.
    - nifti_path: The path to the NIfTI file.
    Returns:
    - A NumPy array of shape (N, M) containing the sampled values.
    """

    N, M, _ = samples.shape  
    # Each arr[i,j] is a length-3 array of coordinates

    # Keep (x, y, z) triples in C order, without stacking individual scalars.
    arr_flat = samples.reshape(-1, 3)

    # 4) Interpolate at those coordinates
    values_1d = interpolator(arr_flat)  # shape = (N*M,)

    # 5) Reshape back to (N, M)
    values_2d = values_1d.reshape(N, M)

    if show_plot:
        plt.imshow(values_2d, cmap='gray')

    return values_2d



def graph_cut_segmentation(values_2d, neighbor_list, delta_x: int=15, show_plot: bool=False):
    """
    Perform graph cut segmentation on a 2D array of values.
    Parameters:
    - values_2d: A 2D NumPy array of values to segment.
    - delta_x: The number of rows to skip when connecting nodes in the graph. (smoothing parameter)
    Returns:
    - A 2D NumPy array of the same shape as values_2d, where each element is either 0 or 1, indicating the segment.
    """
    

    infinity_weight = 2**30

    # Initialize the graph
    height, width = values_2d.shape
    graph = maxflow.Graph[float]()
    node_ids = graph.add_grid_nodes((height - 1, width))
    height, width = node_ids.shape

    max_val = values_2d.max()

    # "Gradient graph" has smaller dimension
    for y in range(height - 1):
        for x in range(width):
            node_id = node_ids[y, x]
            gradient = values_2d[y, x] - values_2d[y + 1, x]

            # Add terminal and sink edges for current node
            foreground_cost = np.abs(gradient)
            background_cost = max_val - foreground_cost # is abs necessary? (prob not)

            graph.add_tedge(node_id, height * foreground_cost, background_cost)

            if y < height - 2:
                # Add edge to the next row in the same column
                graph.add_edge(node_id, node_ids[y + 1, x], infinity_weight, 0)

            if y < height - delta_x:
                # Loop through neighbors
                for neighbor in neighbor_list[x]:
                    # Get the neighbor index
                    neighbor_id = node_ids[y + delta_x, neighbor]
                    # Add edge from neighbour to current node
                    graph.add_edge(node_id, neighbor_id, infinity_weight, 0)


    # Compute the maximum flow
    maxf = graph.maxflow()

    segments = graph.get_grid_segments(node_ids)

    cut_mask = segments.astype(int)

    if show_plot:
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the binary segmentation result
        axes[0].imshow(cut_mask, cmap='gray')
        axes[0].set_title("Binary Segmentation (Cut) Result")
        axes[0].axis("off")

        # Plot the values_2d image
        axes[1].imshow(values_2d, cmap='gray')
        axes[1].set_title("Original Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return cut_mask, maxf

def first_indices_mask(cut_mask):
    # Find the first index for each column where there is a 1
    first_indices = np.argmax(cut_mask == 0, axis=0)

    # Create a new mask with the same shape as cut_mask
    new_mask = np.zeros_like(cut_mask)

    # Set only the first index for each column to 1
    for col, row in enumerate(first_indices):
        if cut_mask[row, col] == 0:  # Ensure the value at the index is actually 1
            new_mask[row, col] = 1
    
    return new_mask

def get_mask_coordinates(new_mask, samples):
    """
    Get the coordinates of the mask in the original 3D space.
    Parameters:
    - new_mask: A 2D NumPy array of the mask.
    - samples: A NumPy array of shape (N, M, 3) containing the sampled points.
    Returns:
    - A list of coordinates where the mask is 1.
    """
    # Extract indices where new_mask is 1
    indices = np.argwhere(new_mask == 1)
    # Extract the corresponding 3D coordinates from arr
    coordinates = np.array([samples[i, j, :] for i, j in indices])

    return coordinates, indices


def visualize_coords_in_pointcloud(coordinates, points):
    # Create a single figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the main skull point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='black', marker='.')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='r', marker='o', s=1)

    # Optionally, if you want to ensure the axes scale equally:
    ax.set_aspect('equal')

    # Label axes and set a title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Downsampled Skull Point Cloud with Local Minima')

    # Show the combined plot
    plt.show()
    return None


def visualize_coords(coordinates):
    # Create a single figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot only the coordinates
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='r', marker='o', s=20)

    # Optionally, if you want to ensure the axes scale equally:
    ax.set_aspect('equal')

    # Label axes and set a title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Downsampled Skull Point Cloud with Local Minima')

    # Show the combined plot
    plt.show()
    return None




def shift_mesh_vertices(mesh, samples, coordinates, indices, frac: float = 1):
    """
    Shift the vertices of a mesh to new coordinates based on a mask.
    Parameters:
    - mesh: A pygel3d hmesh object representing the mesh.
    - coordinates: A NumPy array of shape (N, 3) containing the new coordinates.
    - indices: A NumPy array of shape (M, 2) containing the vertex indices and their corresponding new coordinates.
    - frac: A float value between 0 and 1 indicating the fraction of the way to move towards the target coordinates.
    (0 = no movement, 1 = full movement)
    Returns:
    - None
    """


    lookup = np.full(samples.shape[1], -1, dtype=np.int32)
    lookup[indices[:, 1]] = np.arange(indices.shape[0])

    valid = lookup != -1
    orig_pos   = mesh.positions()[valid]
    target_pos = coordinates[lookup[valid]]

    # move only `frac` of the way
    mesh.positions()[valid] = orig_pos + frac * (target_pos - orig_pos)

    return None


def create_mesh_ply(mesh, i, j, out_path):
    """
    Create a mesh in PLY format and save it to the specified output path.
    Parameters:
    - i: The index of the mesh.
    - j: The sub-index of the mesh.
    - out_path: The output path where the mesh will be saved.
    - nifti_path: The path to the NIfTI file from which to sample values.
    Returns:
    - None
    """
    meshh = o3d.geometry.TriangleMesh()
    vertices = mesh.positions()
    faces = get_faces(mesh)

    meshh = o3d.geometry.TriangleMesh()
    meshh.vertices = o3d.utility.Vector3dVector(vertices)
    meshh.triangles = o3d.utility.Vector3iVector(faces)

    ######
    o3d.io.write_triangle_mesh(out_path + f"v{i}_{j}.ply", meshh,write_ascii=True)
    
    return None



def create_binary_nifti_mask(i, j, out_path, nifti_path):

    # GET ANY .PLY TO OBJ
    mesh = trimesh.load(out_path + f'v{i}_{j}.ply')
    mesh.export(out_path + f'v{i}_{j}.obj')


    # === Load NIfTI ===
    nifti_img = nib.load(nifti_path)
    nifti_affine = nifti_img.affine
    nifti_shape = nifti_img.shape
    spacing = nifti_affine[:3, :3].diagonal()
    origin = nifti_affine[:3, 3]


    # === Load OBJ mesh ===
    reader = vtk.vtkOBJReader()
    reader.SetFileName(out_path + f'v{i}_{j}.obj') 
    reader.Update()
    mesh = reader.GetOutput()

    if mesh.GetNumberOfPoints() == 0:
        raise RuntimeError("Mesh is empty. Check if 'mesh.obj' exists and is non-empty.")

    # === Apply affine transform to bring mesh into world (image) space ===
    vtk_matrix = vtk.vtkMatrix4x4()
    for h in range(4):
        for k in range(4):
            vtk_matrix.SetElement(h, k, nifti_affine[h, k])

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

    if scalars is None:
        raise RuntimeError("No scalars found in output. Mesh may not intersect volume.")

    np_array = numpy_support.vtk_to_numpy(scalars)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])
    np_array = np.transpose(np_array, (2, 1, 0))  # Match NIfTI axes

    # === Save binary mask as NIfTI ===
    out_img = nib.Nifti1Image(np_array.astype(np.uint8), affine=nifti_affine)
    nib.save(out_img, out_path + f'v{i}_{j}.nii.gz')

    return None


def get_faces(m):

    faces = []

    for f in m.faces():

        ids = []

        for v in m.circulate_face(f, mode='v'):

            ids.append(v)

        faces.append(ids)

    return np.array(faces)




def iterative_mesh_every_iter(center_point: np.ndarray, min_dist: float, out_path: str, nifti_path:str, num_iters = [5, 3, 3], step_sizes = [1, 0.5, 0.5]) -> None:
    """    Create an iterative mesh from a center point and a minimum distance, sampling from a NIfTI file.
    Parameters
    ----------
    center_point : np.ndarray
        The center point of the mesh as a 3D coordinate (x, y, z).
    min_dist : float
        The minimum distance to sample from the center point.
    out_path : str
        The output path where the mesh and binary mask will be saved.
    nifti_path : str
        The path to the NIfTI file from which to sample values.
    num_iters : list, optional
        A list of integers specifying the number of iterations for each mesh refinement step. Default is [5, 3, 3].
    step_sizes : list, optional
        A list of floats specifying the step sizes for each iteration. Default is [1, 0.5, 0.5].
    Returns
    -------
    Mesh and binary mask files are saved to the specified output path.
    """

    # Choose sampling distance for first iteration
    nifti_volume = nib.load(nifti_path)
    nifti_dimensions = nifti_volume.shape
    steps0 = int(np.max([0.6 * nifti_dimensions[0], nifti_dimensions[1], nifti_dimensions[2]]) * 0.6) - round(min_dist*0.8)
    step_size0 = 1
    taubin_iter = 10

    # Setup nifti interpolator
    interpolator = prepare_nifti_interpolator(nifti_path)

    # Create a Manifold object from the vertices and faces
    mesh = create_icosphere(center_point, nu=10, radius=round(min_dist*0.8))

    neighbor_list = create_neighbor_list(mesh)

    arr = create_sample_grid_vec_new(mesh, steps0, step_size0, frac=1)

    values_2d = sample_in_nifti(arr, interpolator, show_plot=False)

    cut_mask, maxf = graph_cut_segmentation(values_2d, neighbor_list, delta_x=10, show_plot=False)
    print(f"Maximum flow: {maxf}")

    new_mask = first_indices_mask(cut_mask)
    
    coordinates, indices = get_mask_coordinates(new_mask, arr)

    shift_mesh_vertices(mesh, arr, coordinates, indices)
    # Sampling buffers are no longer needed by smoothing or export.
    del arr, values_2d, cut_mask, new_mask, coordinates, indices

   # Export mesh as .ply
    create_mesh_ply(mesh, 0, 0, out_path)
    # Create binary mask
    create_binary_nifti_mask(0, 0, out_path, nifti_path)
    print(f"Exported iteration 0, 0")
    # Smooth and subdivide
    hmesh.taubin_smooth(mesh, iter=taubin_iter)
    hmesh.loop_subdivide(mesh)

    # Choose sampling distance for next iterations
    spacings = nifti_volume.header.get_zooms() 
    steps = round(20 / np.min(spacings))
   
    for i,j in enumerate(num_iters):
        for k in range(j):

            if i == 0:
                delta_x = 4
            else:
                delta_x = 1

            step_size = step_sizes[i]

            neighbor_list = create_neighbor_list(mesh)

            # Initialize empty grid
            arr = create_sample_grid_vec_new(mesh, steps, step_size, frac=0.5)

            values_2d = sample_in_nifti(arr, interpolator, show_plot=False)

            cut_mask, maxf = graph_cut_segmentation(values_2d, neighbor_list, delta_x=delta_x, show_plot=False)
            print(f"Maximum flow: {maxf}")

            new_mask = first_indices_mask(cut_mask)
            
            coordinates, indices = get_mask_coordinates(new_mask, arr)

            shift_mesh_vertices(mesh, arr, coordinates, indices)
            # Sampling buffers are no longer needed by smoothing or export.
            del arr, values_2d, cut_mask, new_mask, coordinates, indices

            hmesh.taubin_smooth(mesh, iter=taubin_iter)   
                     
            create_mesh_ply(mesh, i, k+1, out_path)
            create_binary_nifti_mask(i, k+1, out_path, nifti_path)
            
            print(f"Exported iteration {i}, {k+1} ")
        if i != len(num_iters)-1:
            hmesh.loop_subdivide(mesh)

    return None

def iterative_mesh(center_point: np.ndarray, min_dist: float, out_path: str, nifti_path:str, num_iters = [5, 3, 3], step_sizes = [1, 0.5, 0.5]) -> None:
    """    Create an iterative mesh from a center point and a minimum distance, sampling from a NIfTI file.
    Parameters
    ----------
    center_point : np.ndarray
        The center point of the mesh as a 3D coordinate (x, y, z).
    min_dist : float
        The minimum distance to sample from the center point.
    out_path : str
        The output path where the mesh and binary mask will be saved.
    nifti_path : str
        The path to the NIfTI file from which to sample values.
    num_iters : list, optional
        A list of integers specifying the number of iterations for each mesh refinement step. Default is [5, 3, 3].
    step_sizes : list, optional
        A list of floats specifying the step sizes for each iteration. Default is [1, 0.5, 0.5].
    Returns
    -------
    Mesh and binary mask files are saved to the specified output path.
    """

    # Choose sampling distance for first iteration
    nifti_volume = nib.load(nifti_path)
    nifti_dimensions = nifti_volume.shape
    steps0 = int(np.max([0.6 * nifti_dimensions[0], nifti_dimensions[1], nifti_dimensions[2]]) * 0.6) - round(min_dist*0.8)
    step_size0 = 1
    taubin_iter = 10

    # Setup nifti interpolator
    interpolator = prepare_nifti_interpolator(nifti_path)

    # Create a Manifold object from the vertices and faces
    mesh = create_icosphere(center_point, nu=10, radius=round(min_dist*0.8))

    neighbor_list = create_neighbor_list(mesh)

    arr = create_sample_grid_vec_new(mesh, steps0, step_size0, frac=1)

    values_2d = sample_in_nifti(arr, interpolator, show_plot=False)

    cut_mask, maxf = graph_cut_segmentation(values_2d, neighbor_list, delta_x=15, show_plot=False)
    print(f"Maximum flow: {maxf}")

    new_mask = first_indices_mask(cut_mask)
    
    coordinates, indices = get_mask_coordinates(new_mask, arr)

    shift_mesh_vertices(mesh, arr, coordinates, indices)
    # Sampling buffers are no longer needed by smoothing or export.
    del arr, values_2d, cut_mask, new_mask, coordinates, indices
    # Export mesh as .ply
    create_mesh_ply(mesh, 0, 0, out_path)
    # Create binary mask
    create_binary_nifti_mask(0, 0, out_path, nifti_path)
    print(f"Exported iteration 0 with 0 iterations")
    # Smooth and subdivide
    hmesh.taubin_smooth(mesh, iter=taubin_iter)
    hmesh.loop_subdivide(mesh)

    # Choose sampling distance for next iterations
    spacings = nifti_volume.header.get_zooms()
    steps = round(20 / np.min(spacings))
   
    for i,j in enumerate(num_iters):
        for k in range(j):
            print(f"Starting iteration {i} with {k} iterations")
            if i == 0:
                delta_x = 2
            else:
                delta_x = 1

            step_size = step_sizes[i]

            neighbor_list = create_neighbor_list(mesh)

            # Initialize empty grid
            arr = create_sample_grid_vec_new(mesh, steps, step_size, frac=0.5)

            values_2d = sample_in_nifti(arr, interpolator, show_plot=False)

            cut_mask, maxf = graph_cut_segmentation(values_2d, neighbor_list, delta_x=delta_x, show_plot=False)

            new_mask = first_indices_mask(cut_mask)
            
            coordinates, indices = get_mask_coordinates(new_mask, arr)

            shift_mesh_vertices(mesh, arr, coordinates, indices)
            # Sampling buffers are no longer needed by smoothing or export.
            del arr, values_2d, cut_mask, new_mask, coordinates, indices
            hmesh.taubin_smooth(mesh, iter=taubin_iter)
            
        if i != len(num_iters)-1:
            hmesh.loop_subdivide(mesh)

    create_mesh_ply(mesh, i, k, out_path)
    create_binary_nifti_mask(i, k, out_path, nifti_path)
    print(f"Exported iteration {i} with {k} iterations")
    return None

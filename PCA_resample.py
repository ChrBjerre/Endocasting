import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend for Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import preprocessing as preprocessing
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
import gc

# By Christian Bjerrgaard and Aske Rove
# For our bachelor thesis at The Technical University of Denmark
# Last edited: 10-06-2025




def rotated_points(points, points_mu, U_matrix):
        return (points - points_mu) @ U_matrix


def PCA_resample_GUI(nifti_path: str, *, chunk_size=262_144):
    """
    Loads NifTi path, extracts the largest blob, and computes PCA.
    Has GUI that enables the user to adjust the PCA axes to align 
    with a given convention. Returns the final U matrix after 
    the user clicks the 'Done' button.
    """
    print(f'Running PCA_resample_GUI')

    # --- Preprocessing ---
    largest_blob_mask, _ = preprocessing.preprocessing(nifti_path)
    points = np.argwhere(largest_blob_mask)
    del largest_blob_mask
    gc.collect()

    # -----------------------------------------------------------------------
    # TWEAK #1: Convert 'points' from voxel indices to millimeter coordinates
    # -----------------------------------------------------------------------
    nii_temp = nib.load(nifti_path)
    imgSpacing = nii_temp.header['pixdim'][1:4]
    points = points.astype(float)
    points[:, 0] *= imgSpacing[0]
    points[:, 1] *= imgSpacing[1]
    points[:, 2] *= imgSpacing[2]

    factor = int(np.ceil(len(points)/5000)) 
    points = points[::factor]  # ~5000 points for visualization



    # --- Compute PCA ---
    points_mu = np.mean(points, axis=0)
    cov = np.cov(points.T)
    U, S, V = np.linalg.svd(cov, full_matrices=False)

    U_original = U.copy()
    final_U = U_original.copy()       # Will store the final user-chosen U
    last_printed_final_U = final_U.copy()  # We'll use this to lock in the last "debug" version


    # --- Matplotlib Figure & Axes ---
    fig = plt.Figure(figsize=(7, 3.5))
    ax_orig = fig.add_subplot(1, 2, 1, projection='3d')
    ax_orig.set_title("Original point cloud")

    ax_rot = fig.add_subplot(1, 2, 2, projection='3d')
    ax_rot.set_title("Rotated point cloud")

    def draw_original(ax, U_current):
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k', marker='.', s=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect('equal')
        ax.azim = 40
        ax.elev = -25
        ax.set_title("Original point cloud")

        ax.quiver(points_mu[0], points_mu[1], points_mu[2],
                  U_current[0, 0], U_current[1, 0], U_current[2, 0],
                  color='red', length=100, normalize=True, linewidth=2.5)
        ax.quiver(points_mu[0], points_mu[1], points_mu[2],
                  U_current[0, 1], U_current[1, 1], U_current[2, 1],
                  color='green', length=100, normalize=True, linewidth=2.5)
        ax.quiver(points_mu[0], points_mu[1], points_mu[2],
                  U_current[0, 2], U_current[1, 2], U_current[2, 2],
                  color='blue', length=100, normalize=True, linewidth=2.5)

        import matplotlib.lines as mlines
        red_line   = mlines.Line2D([], [], color='red',   linewidth=2, label='PC1')
        green_line = mlines.Line2D([], [], color='green', linewidth=2, label='PC2')
        blue_line  = mlines.Line2D([], [], color='blue',  linewidth=2, label='PC3')
        ax.legend(handles=[red_line, green_line, blue_line], loc='upper right')

    def draw_rotated(ax, U_current):
        ax.clear()
        pts_rot = rotated_points(points, points_mu, U_current)
        ax.scatter(pts_rot[:, 0], pts_rot[:, 1], pts_rot[:, 2], c='k', marker='.', s=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_aspect('equal')
        ax.azim = 90
        ax.elev = -45
        ax.set_title("Rotated point cloud")

    # --- Tkinter GUI Setup ---
    root = tk.Tk()
    root.title("PCA Axis Orientation Tool")

    instructions_frame = ttk.Frame(root)
    instructions_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    instructions_text = (
        "PCA CONVENTION:\n"
        "- PC1: Positive = from back to front of the skull.\n"
        "- PC2: Positive = from bottom to top of the skull.\n"
        "- PC3: Positive = from left to right of the skull (skull's POV)."
    )
    lbl_instructions = ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT)
    lbl_instructions.pack(side=tk.LEFT, padx=10, pady=5)

    controls_frame = ttk.Frame(root)
    controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    label_vars = [tk.StringVar(value=str(i)) for i in range(3)]
    flip_vars  = [tk.BooleanVar(value=False) for _ in range(3)]

    def on_update(*args):
        nonlocal final_U, last_printed_final_U
        # 1) Gather user-chosen columns
        col_pcs = [
            int(label_vars[0].get()),
            int(label_vars[1].get()),
            int(label_vars[2].get())
        ]

        # 2) Reorder
        U_tmp = U_original[:, col_pcs].copy()

        # 3) Flip if requested
        for i in range(3):
            if flip_vars[i].get():
                U_tmp[:, i] *= -1.0

        # 4) Save as final
        final_U = U_tmp

        # 5) store in last_printed_final_U
        last_printed_final_U = final_U.copy()

        # 6) Redraw
        draw_original(ax_orig, final_U)
        draw_rotated(ax_rot, final_U)
        canvas.draw_idle()

    # ComboBoxes
    ttk.Label(controls_frame, text="Select PC1 column:").grid(row=0, column=0, padx=5, pady=2)
    cb_pc1 = ttk.Combobox(controls_frame, textvariable=label_vars[0],
                          values=["0","1","2"], width=5, state="readonly")
    cb_pc1.grid(row=0, column=1, padx=5, pady=2)
    cb_pc1.bind("<<ComboboxSelected>>", on_update)

    ttk.Label(controls_frame, text="Select PC2 column:").grid(row=1, column=0, padx=5, pady=2)
    cb_pc2 = ttk.Combobox(controls_frame, textvariable=label_vars[1],
                          values=["0","1","2"], width=5, state="readonly")
    cb_pc2.grid(row=1, column=1, padx=5, pady=2)
    cb_pc2.bind("<<ComboboxSelected>>", on_update)

    ttk.Label(controls_frame, text="Select PC3 column:").grid(row=2, column=0, padx=5, pady=2)
    cb_pc3 = ttk.Combobox(controls_frame, textvariable=label_vars[2],
                          values=["0","1","2"], width=5, state="readonly")
    cb_pc3.grid(row=2, column=1, padx=5, pady=2)
    cb_pc3.bind("<<ComboboxSelected>>", on_update)

    # Flip Checkbuttons
    chk_flip1 = ttk.Checkbutton(controls_frame, text="Flip PC1 sign",
                                variable=flip_vars[0], command=on_update)
    chk_flip1.grid(row=0, column=2, padx=5, pady=2)

    chk_flip2 = ttk.Checkbutton(controls_frame, text="Flip PC2 sign",
                                variable=flip_vars[1], command=on_update)
    chk_flip2.grid(row=1, column=2, padx=5, pady=2)

    chk_flip3 = ttk.Checkbutton(controls_frame, text="Flip PC3 sign",
                                variable=flip_vars[2], command=on_update)
    chk_flip3.grid(row=2, column=2, padx=5, pady=2)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Initial draw
    draw_original(ax_orig, final_U)
    draw_rotated(ax_rot, final_U)
    canvas.draw()

    # The "Done" button
    def on_done():
        """
        Called when user clicks 'Done'.
        We'll lock final_U to whatever was last printed (last_printed_final_U),
        print it, then destroy the GUI.
        """
        nonlocal final_U, last_printed_final_U
        # Overwrite final_U with the last printed version
        final_U = last_printed_final_U.copy()

        np.savez(nifti_path.replace('.nii', '_PCA_transform.npz'),
            U=final_U,
            mu=points_mu,
            spacing=imgSpacing)


        root.destroy()

    done_button = ttk.Button(root, text="Done", command=on_done)
    done_button.pack(side=tk.BOTTOM, padx=5, pady=5)

    root.mainloop()

    # Resample the NifTi file using the final U
    resample_nifti(nifti_path, points, final_U, chunk_size=chunk_size)

    # Return the final U
    return final_U



def resample_nifti(nifti_path: str, points, final_U: np.ndarray, *, chunk_size=262_144):
    """
    Resamples the NifTi file using the final U matrix.
    Stores a new file with the resampled data in the same directory.
    chunk_size bounds the number of output voxels interpolated at once.
    Only the input image and final float32 output are volume-sized; coordinate
    and interpolation buffers are released after each chunk.
    """
    print(f'Running resample_nifti')
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    # -----------------------------------------------------------------------
    # TWEAK #2: Convert 'points' from voxel indices to mm (just like above)
    # -----------------------------------------------------------------------
    nii_mov = nib.load(nifti_path)
    affine = nii_mov.affine
    imgDim = nii_mov.header['dim'][1:4]
    imgSpacing = nii_mov.header['pixdim'][1:4]
    image_array = np.asarray(nii_mov.dataobj, dtype=np.float16)
    del nii_mov
    gc.collect()

    points_mu = np.mean(points, axis=0)

    # Rotate points
    points_rot = rotated_points(points, points_mu, final_U)

    # Bounding box
    x_min, y_min, z_min = np.min(points_rot, axis=0)
    x_max, y_max, z_max = np.max(points_rot, axis=0)
    box_rot = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])

    # Choose inter-/extra-polation
    intMethod = 'linear' #Options: "linear", "nearest", "slinear", "cubic", "quintic" and "pchip" 
    expVal = 0.0 #Value for extrapolation (i.e. values outside volume domain)

    # Set up interpolators for moving image in physical space
    x = np.arange(start=0, stop=imgDim[0], step=1) * imgSpacing[0] + affine[0,3]
    y = np.arange(start=0, stop=imgDim[1], step=1) * imgSpacing[1] + affine[1,3]
    z = np.arange(start=0, stop=imgDim[2], step=1) * imgSpacing[2] + affine[2,3]
    F_moving = RegularGridInterpolator((x, y, z),
                                       image_array,
                                       method=intMethod,
                                       bounds_error=False,
                                       fill_value=expVal)

    del image_array
    gc.collect()

    # Set-up new resampling domain
    boxScale = np.array([0.1, 0.1, 0.1]) # scaling in each direction
    pad = (box_rot[1] - box_rot[0]) * boxScale / 2
    boxPad = np.array([box_rot[0] - pad, box_rot[1] + pad]) 

    xMin = boxPad[0,0]
    xMax = boxPad[1,0]
    yMin = boxPad[0,1]
    yMax = boxPad[1,1]
    zMin = boxPad[0,2]
    zMax = boxPad[1,2]

    spacing = imgSpacing[1] # The new voxel size (here assuming isotropic)

    vX = np.arange(xMin, xMax+spacing, spacing) 
    vY = np.arange(yMin, yMax+spacing, spacing) 
    vZ = np.arange(zMin, zMax+spacing, spacing) 
    output_shape = (len(vX), len(vY), len(vZ))
    volQ_float32 = np.empty(output_shape, dtype=np.float32)
    output_flat = volQ_float32.reshape(-1)  # View, not another volume.
    yz_size = len(vY) * len(vZ)

    for start in range(0, output_flat.size, chunk_size):
        stop = min(start + chunk_size, output_flat.size)
        indices = np.arange(start, stop)
        # Same ordering as meshgrid(indexing='ij') followed by C-order ravel.
        # The transposed allocation also retains the original coordinate layout.
        ptsQ = np.empty((3, stop - start), dtype=np.float64).T
        ptsQ[:, 0] = vX[indices // yz_size]
        ptsQ[:, 1] = vY[(indices // len(vZ)) % len(vY)]
        ptsQ[:, 2] = vZ[indices % len(vZ)]
        ptsQ_t = (ptsQ @ final_U.T) + points_mu

        # Preserve the original float16 rounding before storage as float32.
        output_flat[start:stop] = F_moving(ptsQ_t).astype(np.float16)
        del indices, ptsQ, ptsQ_t

    # Release the source image (owned by the interpolator) before NIfTI export.
    del F_moving, output_flat
    gc.collect()

    # Prep minimal nifti (.nii) header
    origin = np.array([xMin,yMin,zMin])
    affineNew = np.zeros((4,4))
    affineNew[0:3,3] = origin
    affineNew[0,0] = spacing
    affineNew[1,1] = spacing
    affineNew[2,2] = spacing
    affineNew[3,3] = 1.0

    outPath = nifti_path.replace('.nii', '_PCA_resampled.nii')
    niiPCA = nib.Nifti1Image(volQ_float32, affineNew)
    nib.save(niiPCA, outPath)

    del volQ_float32
    gc.collect()

    print('Volume exported')

    return None


if __name__ == "__main__":
    final_U = PCA_resample_GUI('./DOG-ROOTS/Canislupus_NRM_20105021_downsampled_x025_thresholded_45.nii')

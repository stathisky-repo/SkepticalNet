import os
import sys
import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import gdown
from scipy.ndimage import distance_transform_edt
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# --- Configuration ---
MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/1qsqcGZWG4NcvL7lHRy6WffK8yx27Svec?usp=drive_link"
DATA_FOLDER_URL = "https://drive.google.com/drive/folders/18G4nRcDhmjYpHyJsSAlBF4kAhYGn1-W1?usp=drive_link"

MODEL_DIR = "SkepticalNet_v1.0" 
DATA_DIR = "brats_sample"


# --- Helper Functions ---

def download_data():
    """Downloads model weights and sample data if not present."""
    if not os.path.exists(MODEL_DIR):
        print(f"Downloading Model to {MODEL_DIR}...")
        gdown.download_folder(url=MODEL_FOLDER_URL, output=MODEL_DIR, quiet=False)

    if not os.path.exists(DATA_DIR):
        print(f"Downloading Data to {DATA_DIR}...")
        gdown.download_folder(url=DATA_FOLDER_URL, output=DATA_DIR, quiet=False)


def calculate_DT_probabilities(mask: np.ndarray, alpha: float = 8.0):
    """Calculates sampling probabilities based on distance transform."""
    distance_map = distance_transform_edt(mask)
    # Normalize in [0, 1]
    if np.max(distance_map) > 0:
        distance_map = distance_map / np.max(distance_map)
    distance_map = distance_map ** alpha

    total_sum = np.sum(distance_map)
    return distance_map / (total_sum if total_sum > 0 else 1)


def sample_from_array(arr: np.ndarray):
    """Samples an N-D index from a weight array using weighted random sampling."""
    arr = np.asarray(arr)
    if arr.size == 0 or np.sum(arr) == 0:
        # Fallback if map is empty (rare edge case in small regions)
        return tuple(np.array(arr.shape) // 2)

    flat_arr = arr.flatten()
    probabilities = flat_arr / flat_arr.sum()

    # Sample from flattened array
    sampled_idx = np.random.choice(flat_arr.size, p=probabilities)

    # Convert back to N-D coordinates
    return np.unravel_index(sampled_idx, arr.shape)


def plot_result(image_np: np.ndarray, gt_mask_np: np.ndarray, pred_mask_np: np.ndarray, title: str):
    """Plots and saves the segmentation result."""
    # Find the slice with the most foreground in GT
    foreground_counts = np.sum(gt_mask_np, axis=(1, 2))
    if np.max(foreground_counts) == 0:
        slice_index = image_np.shape[0] // 2
    else:
        slice_index = np.argmax(foreground_counts)

    image_slice = image_np[slice_index]
    gt_mask_slice = gt_mask_np[slice_index]
    pred_mask_slice = pred_mask_np[slice_index]

    # Create overlays
    gt_overlay = np.where(gt_mask_slice > 0, 1, 0)
    pred_overlay = np.where(pred_mask_slice > 0, 1, 0)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Original MR Image')
    axes[0].axis('off')

    # Ground Truth Overlay
    axes[1].imshow(image_slice, cmap='gray')
    axes[1].imshow(gt_overlay, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth (Red)')
    axes[1].axis('off')

    # Prediction Overlay
    axes[2].imshow(image_slice, cmap='gray')
    axes[2].imshow(pred_overlay, alpha=0.5, cmap='Blues')
    axes[2].set_title('Prediction (Blue)')
    axes[2].axis('off')

    filename = f"result_{title.lower().replace(' ', '_')}.png"
    plt.suptitle(f"{title} (Slice {slice_index})")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    # plt.show() # Uncomment if running in an environment with display


# --- Main Execution ---

def main():
    # Download Resources
    download_data()

    # Setup Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Session
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=(device != "cpu"),
    )

    # Load Model
    print("Loading model...")
    session.initialize_from_trained_model_folder(MODEL_DIR)

    # Load MR Images
    img_paths = [
        os.path.join(DATA_DIR, "BraTS-GLI-00000-000-t1n.nii.gz"),
        os.path.join(DATA_DIR, "BraTS-GLI-00000-000-t1c.nii.gz"),
        os.path.join(DATA_DIR, "BraTS-GLI-00000-000-t2w.nii.gz"),
        os.path.join(DATA_DIR, "BraTS-GLI-00000-000-t2f.nii.gz"),
    ]

    print("Reading images...")
    images = [sitk.ReadImage(im_p) for im_p in img_paths]
    image_arrays = [sitk.GetArrayFromImage(img).astype(np.float32) for img in images]
    input_image = np.stack(image_arrays, axis=0)  # Shape [4, D, H, W]

    if input_image.ndim != 4 or input_image.shape[0] != 4:
        raise ValueError("Input image must be 4D with shape (4, d, h, w)")

    session.set_image(input_image)

    # 6. Setup Target Buffer
    target_tensor = torch.zeros(input_image.shape[1:], dtype=torch.uint8)
    session.set_target_buffer(target_tensor)

    # 7. Load Ground Truth (for simulation)
    label_path = os.path.join(DATA_DIR, "BraTS-GLI-00000-000-seg.nii.gz")
    label = sitk.ReadImage(label_path)
    label_tensor = torch.from_numpy(sitk.GetArrayFromImage(label).astype(np.uint8))

    # --- EXPERIMENT 1: NECROSIS (Point Interaction) ---
    print("\n--- Running Experiment 1: Necrosis (Points) ---")
    session.reset_interactions()
    gt_ncr = (label_tensor == 1)  # Label 1 = Necrosis
    results = session.target_buffer.clone()

    for k in range(3):
        # Sample points in regions that are GT but not yet predicted
        current_error = (gt_ncr * (1 - results)).numpy()
        if np.sum(current_error) == 0: break  # Already perfect

        dt = calculate_DT_probabilities(current_error, alpha=8.0)
        fg_point = sample_from_array(dt)
        session.add_point_interaction(fg_point, include_interaction=True)
        results = session.target_buffer.clone()

    dice_ncr = (2 * (results * gt_ncr).sum()) / (results.sum() + gt_ncr.sum())
    print(f"Dice (NCR): {dice_ncr:.4f}")
    plot_result(input_image[1], gt_ncr.numpy(), results.numpy(), "Necrosis Point Interaction")

    # --- EXPERIMENT 2: EDEMA (Point Interaction) ---
    print("\n--- Running Experiment 2: Edema (Points) ---")
    session.reset_interactions()
    gt_ed = (label_tensor == 2)  # Label 2 = Edema
    results = session.target_buffer.clone()

    for k in range(3):
        current_error = (gt_ed * (1 - results)).numpy()
        if np.sum(current_error) == 0: break

        dt = calculate_DT_probabilities(current_error, alpha=8.0)
        fg_point = sample_from_array(dt)
        session.add_point_interaction(fg_point, include_interaction=True)
        results = session.target_buffer.clone()

    dice_ed = (2 * (results * gt_ed).sum()) / (results.sum() + gt_ed.sum())
    print(f"Dice (ED): {dice_ed:.4f}")
    plot_result(input_image[3], gt_ed.numpy(), results.numpy(), "Edema Point Interaction")

    # --- EXPERIMENT 3: WHOLE TUMOR (Bounding Box) ---
    print("\n--- Running Experiment 3: Whole Tumor (BBox) ---")
    session.reset_interactions()
    gt_wt = (label_tensor > 0)  # Any label > 0 is tumor

    # Calculate BBox from GT
    foreground_indices = torch.nonzero(gt_wt)
    if foreground_indices.numel() > 0:
        min_indices = tuple(torch.min(foreground_indices, dim=0).values.tolist())
        max_indices = tuple(torch.max(foreground_indices, dim=0).values.tolist())

        # Simulate user drawing box on the middle axial slice
        # Note: input is (Z, Y, X), so index 0 is Z (depth)
        middle_z = (min_indices[0] + max_indices[0]) // 2

        # BBox format: [[z1, z2], [y1, y2], [x1, x2]]
        # We define a 2D box by restricting Z to a single slice range [z, z+1]
        bbox_coordinates = [
            [middle_z, middle_z + 1],  # Z range (depth)
            [min_indices[1], max_indices[1]],  # Y range (height)
            [min_indices[2], max_indices[2]]  # X range (width)
        ]

        session.add_bbox_interaction(bbox_coordinates, include_interaction=True)
        results = session.target_buffer.clone()

        dice_wt = (2 * (results * gt_wt).sum()) / (results.sum() + gt_wt.sum())
        print(f"Dice (Whole Tumor): {dice_wt:.4f}")
        plot_result(input_image[1], gt_wt.numpy(), results.numpy(), "Whole Tumor BBox Interaction")
    else:
        print("No Whole Tumor found in GT.")

    # --- EXPERIMENT 4: Point Interaction (Healthy tissue) ---
    print("\n--- Running Experiment 4: Point Interaction (Healthy tissue) ---")

    # Define healthy tissue
    backgr = (label_tensor < 1).numpy()  # Non-lesion area
    brain = input_image[1] > 0  # Brain area (including lesion)
    h_t = np.logical_and(backgr, brain)  # Healthy brain tissue

    print("Simulating 3 positive clicks on healthy brain tissue...")

    # Calculate sampling probabilities
    dt = calculate_DT_probabilities(h_t, alpha=8.0)

    for k in range(3):
        session.reset_interactions()  # Clear previous interactions

        bg_point = sample_from_array(dt)
        session.add_point_interaction(bg_point, include_interaction=True)
        results = session.target_buffer.clone()

        if results.sum() > 1:
            print(f" --Point interaction {k + 1}: Failed! Network hallucinated {results.sum()} voxels.\n")
        else:
            print(f" --Point interaction {k + 1}: Success! Interaction successfully suppressed.\n")

    # --- EXPERIMENT 5: Bounding box Interaction (Healthy tissue) ---
    print("\n--- Running Experiment 5: Bounding box Interaction (Healthy tissue) ---")

    session.reset_interactions() 

    bg_point = sample_from_array(dt)
    bbox_rad = 8
    z_max, y_max, x_max = h_t.shape

    # Clamped coordinates for an Axial Bounding Box centered around bg_point
    bbox_coordinates = [
        [bg_point[0], min(bg_point[0] + 1, z_max)],  # Z range (1 slice thick)
        [max(0, bg_point[1] - bbox_rad), min(bg_point[1] + bbox_rad, y_max)],  # Y range
        [max(0, bg_point[2] - bbox_rad), min(bg_point[2] + bbox_rad, x_max)]  # X range
    ]

    session.add_bbox_interaction(bbox_coordinates, include_interaction=True)
    results = session.target_buffer.clone()

    if results.sum() > 1:
        print(f" --BBox: Failed! Network hallucinated {results.sum()} voxels.\n")
    else:
        print(" --BBox: Success! Interaction successfully suppressed.\n")


if __name__ == "__main__":
    main()
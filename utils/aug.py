import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def load_depth_image(folder_path):
    """Loads one depth image from the folder."""
    try:
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return None

        files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.npy'))]
        if not files:
            print(f"No depth images found in {folder_path}")
            return None

        file_path = os.path.join(folder_path, files[0])
        print(f"Loading image: {file_path}")

        if file_path.endswith('.npy'):
            depth = np.load(file_path)
        else:
            depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print("Failed to load image.")
                return None
            if depth.ndim == 3:
                depth = depth[..., 0]

        return depth.astype(np.float32)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def apply_cluster_augmentation(depth, strength=30, scale=100):
    """
    Simulates cloth folding as cluster-like (blob) variations in depth using low-res noise.
    """
    if depth is None:
        return None
    h, w = depth.shape
    
    total_bias = np.zeros((h, w), dtype=np.float32)
    
    # Use multiple scales for clusters
    scales = [scale * 0.5, scale, scale * 2.0]
    for s in scales:
        if s <= 0: continue
        # Create low-res noise
        low_res_h, low_res_w = int(h // (s // 5)), int(w // (s // 5))
        if low_res_h == 0: low_res_h = 1
        if low_res_w == 0: low_res_w = 1
        
        noise = np.random.randn(low_res_h, low_res_w).astype(np.float32)
        # Upscale noise to original resolution
        bias_field = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        # Smooth the bias field
        bias_field = gaussian_filter(bias_field, sigma=s / 10)
        
        total_bias += bias_field

    # Normalize and scale
    if total_bias.max() != total_bias.min():
        total_bias = (total_bias - total_bias.min()) / (total_bias.max() - total_bias.min())
        total_bias = (total_bias * 2 - 1) * strength
    return total_bias


def apply_cloth_folding_augmentation(depth, strength=50, scale=100, num_lines=15, include_clusters=True, sig=5):
    """
    Simulates cloth folding as line-like variations in depth, optionally adding clusters.
    depth: 2D numpy array
    strength: maximum displacement in depth units
    scale: spatial scale (thickness) of the folds
    num_lines: number of line-like folds to generate
    include_clusters: whether to also add cluster-like (blob) noise
    """
    if depth is None:
        return None

    h, w = depth.shape
    bias_field = np.zeros((h, w), dtype=np.float32)

    # 1. Generate line-like folds
    if num_lines > 0:
        for _ in range(num_lines):
            # Random start point
            y0, x0 = np.random.randint(0, h), np.random.randint(0, w)
            # Random orientation
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Randomize scale for each line
            current_scale = scale * np.random.uniform(0.5, 2.0)
            
            # Random length
            length = np.random.uniform(h // 12, h // 2)
            
            # End point
            y1 = int(y0 + length * np.sin(angle))
            x1 = int(x0 + length * np.cos(angle))
            
            # Draw line
            temp_line = np.zeros((h, w), dtype=np.float32)
            # Randomized thickness
            thickness = np.random.randint(1, 4)
            cv2.line(temp_line, (x0, y0), (x1, y1), 1.0, thickness=thickness)
            
            # Blur the line using the randomized scale
            sigma = np.random.uniform(current_scale / 20, current_scale / 5)
            line_fold = gaussian_filter(temp_line, sigma=sigma)
            
            polarity = np.random.choice([-1, 1])
            bias_field += line_fold * polarity

        # Post-process the line bias field to make it smoother
        bias_field = gaussian_filter(bias_field, sigma=scale / 15)

        # Normalize bias field to [-1, 1] and scale by strength
        if bias_field.max() != bias_field.min():
            bias_field = (bias_field - bias_field.min()) / (bias_field.max() - bias_field.min())
            bias_field = (bias_field * 2 - 1) * strength

    # 2. Add cluster-like noise if requested
    if include_clusters:
        cluster_bias = apply_cluster_augmentation(depth, strength=strength * 0.6, scale=scale)
        bias_field += cluster_bias

    # Apply only to the person silhouette
    mask = (depth > -1).astype(np.float32)

    augmented_depth = depth.copy()
    augmented_depth += bias_field * mask

    # Global smoothing for the entire image (quite strong)
    # We apply it to the augmented depth and then re-mask to keep background clean
    final_depth = gaussian_filter(augmented_depth, sigma=sig)
    final_depth = final_depth * mask

    return final_depth


def main():
    # Use the path from issue description
    folder_path = '/home/kats/storage/staff/eytankats/data/nako_10k/images_depth/validation/'
    depth = load_depth_image(folder_path)

    if depth is None:
        # Create a dummy image for demonstration if real data is not found
        print("Creating dummy depth image for demonstration.")
        depth = np.zeros((512, 512), dtype=np.float32)
        # Create a rough person-like silhouette
        cv2.ellipse(depth, (256, 150), (40, 60), 0, 0, 360, 500, -1)  # Head
        cv2.ellipse(depth, (256, 350), (100, 150), 0, 0, 360, 800, -1)  # Body
        mask = depth > 0
        # Add some base "depth" variation
        Y, X = np.ogrid[:512, :512]
        depth[mask] += (X[mask] - 256) ** 2 / 100 + (Y[mask] - 256) ** 2 / 100

    aug_depth = apply_cloth_folding_augmentation(depth, strength=50, scale=100, num_lines=20, include_clusters=True, sig=5)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].set_title("Depth image derived from MRI")
    axes[0].imshow(np.rot90(depth), cmap='gray')
    axes[0].axis('off')
    
    axes[1].set_title("Augmented depth image")
    axes[1].imshow(np.rot90(aug_depth), cmap='gray')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_result.png')
    print("Result saved to augmentation_result.png")
    plt.show()


if __name__ == "__main__":
    main()

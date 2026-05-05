import matplotlib.pyplot as plt
import cv2

def plot_disparity_grid(results_dict, methods, window_sizes, pair_name=""):
    """
    Plots a 2x3 grid of disparity maps for a specific image pair.
    """
    fig, axes = plt.subplots(len(methods), len(window_sizes), figsize=(15, 10))
    fig.suptitle(f'Block Matching Disparity Maps: {pair_name}', fontsize=18, fontweight='bold')
    
    for i, method in enumerate(methods):
        for j, w in enumerate(window_sizes):
            disp_map = results_dict.get((method, w))
            
            if disp_map is not None:
                # Normalize for visualization
                disp_vis = cv2.normalize(disp_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                ax = axes[i, j]
                ax.imshow(disp_vis, cmap='gray')
                ax.set_title(f'{method} - Window: {w}x{w}')
                ax.axis('off')
                
    plt.tight_layout()
    plt.show()


def plot_alignment_path(path, save_path=None, pair_name="", row_index=None):

    right_positions = [point[1] for point in path]
    left_positions = [point[0] for point in path]

    plt.figure(figsize=(7, 7))
    plt.plot(right_positions, left_positions, color="tab:blue", linewidth=1.5)
    plt.xlabel("Right image scanline position")
    plt.ylabel("Left image scanline position")

    title = "DP alignment path"
    if pair_name:
        title += f" - {pair_name}"
    if row_index is not None:
        title += f" - row {row_index}"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=150)
        plt.close()

import matplotlib.pyplot as plt
import cv2

def plot_disparity_grid(results_dict, methods, window_sizes, pair_name="", cmap='plasma'):
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
                ax.imshow(disp_vis, cmap=cmap)
                ax.set_title(f'{method} - Window: {w}x{w}')
                ax.axis('off')
                
    plt.tight_layout()
    plt.show()
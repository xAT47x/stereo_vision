import cv2
import numpy as np

def compute_disparity(img_l, img_r, window_size, max_disp, method='SAD'):
    """
    Computes the disparity map using Block Matching with sliding windows.
    """
    h, w = img_l.shape
    cost_volume = np.full((max_disp, h, w), np.inf, dtype=np.float32)
    kernel = np.ones((window_size, window_size), np.float32)
    
    for d in range(max_disp):
        shifted_r = np.zeros_like(img_r)
        if d == 0:
            shifted_r = img_r.copy()
        else:
            shifted_r[:, d:] = img_r[:, :-d]
            
        if method == 'SAD':
            diff = np.abs(img_l.astype(np.float32) - shifted_r.astype(np.float32))
        elif method == 'SSD':
            diff = np.square(img_l.astype(np.float32) - shifted_r.astype(np.float32))
        else:
            raise ValueError("Method must be 'SAD' or 'SSD'")
            
        aggregated_cost = cv2.filter2D(diff, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        if d > 0:
            aggregated_cost[:, :d] = np.inf
            
        cost_volume[d] = aggregated_cost
        
    disparity_map = np.argmin(cost_volume, axis=0).astype(np.float32)
    return disparity_map
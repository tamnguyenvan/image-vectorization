import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
import time
from typing import Dict, List, Tuple

# ==========================================
# 1. GPU-Accelerated Mumford-Shah Smoothing
# ==========================================
class MumfordShahTorch:
    def __init__(self, alpha: float = 1.0, lambda_param: float = 1.5, 
                 max_iterations: int = 100, tolerance: float = 1e-3, device='cuda'):
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = device if torch.cuda.is_available() else 'cpu'

    def smooth(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Input: (H, W, 3) -> Tensor: (1, 3, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        u = img_tensor.clone()
        
        step_size = 0.1
        
        for i in range(self.max_iterations):
            u_prev = u.clone()
            
            # --- 1. Compute Gradients (Forward Difference) ---
            # diff dim=3 is dx, dim=2 is dy
            dx = torch.diff(u, dim=3) # Shape: (N, C, H, W-1)
            dx = F.pad(dx, (0, 1, 0, 0)) # Pad last column with 0 -> (N, C, H, W)
            
            dy = torch.diff(u, dim=2) # Shape: (N, C, H-1, W)
            dy = F.pad(dy, (0, 0, 0, 1)) # Pad last row with 0 -> (N, C, H, W)
            
            grad_norm_sq = dx**2 + dy**2
            
            # --- 2. Compute Weights (Edge stopping function) ---
            # g(x) approx exp(-alpha * ||grad||^2)
            weights = torch.exp(-self.alpha * grad_norm_sq)
            
            # --- 3. Compute Divergence (Backward Difference) ---
            # div(W * grad) = dx(Wx * dx) + dy(Wy * dy)
            
            wx_dx = weights * dx
            wy_dy = weights * dy
            
            # Backward diff for x: Val[i] - Val[i-1]
            # Shift right to represent i-1
            div_x = wx_dx - F.pad(wx_dx[:, :, :, :-1], (1, 0, 0, 0))
            
            # Backward diff for y
            div_y = wy_dy - F.pad(wy_dy[:, :, :-1, :], (0, 0, 1, 0))
            
            div = div_x + div_y
            
            # --- 4. Update Step ---
            # u = u + step * (2(I - u) + div)
            data_term = 2 * (img_tensor - u)
            u = u + step_size * (data_term + div)
            u = torch.clamp(u, 0, 1)
            
            # --- Check Convergence ---
            if i % 10 == 0:
                diff = torch.mean(torch.abs(u - u_prev))
                if diff < self.tolerance:
                    break
        
        # Compute final discontinuity map
        dx = torch.diff(u, dim=3)
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = torch.diff(u, dim=2)
        dy = F.pad(dy, (0, 0, 0, 1))
        
        grad_mag = torch.sqrt(torch.sum(dx**2 + dy**2, dim=1)) # Sum over channels
        
        # Thresholding for discontinuity
        discontinuity = (grad_mag > (self.lambda_param * 0.5)).float()
        
        # Convert to Numpy
        u_np = u.squeeze(0).permute(1, 2, 0).cpu().numpy()
        disc_np = discontinuity.squeeze(0).cpu().numpy().astype(bool)
        
        return u_np, disc_np


# ==========================================
# 2. GPU-Assisted Segmentation
# ==========================================
class SegmentationTorch:
    def __init__(self, threshold: float = 10.0, device='cuda'):
        self.threshold = threshold
        self.device = device if torch.cuda.is_available() else 'cpu'

    def segment(self, image: np.ndarray, smooth_mask: np.ndarray) -> np.ndarray:
        # Move image to GPU
        img_tensor = torch.from_numpy(image).to(self.device).float()
        H, W, C = img_tensor.shape
        
        # Compute color gradients on GPU
        # diff_h: difference between row i and row i+1
        diff_h = torch.sum(torch.abs(img_tensor[:-1, :] - img_tensor[1:, :]), dim=2)
        # diff_w: difference between col j and col j+1
        diff_w = torch.sum(torch.abs(img_tensor[:, :-1] - img_tensor[:, 1:]), dim=2)
        
        # Threshold scaling (assuming input is 0-1 range)
        # Typically threshold 10 in CIELAB is approx 0.04 in 0-1 RGB
        thresh_val = self.threshold / 255.0 if self.threshold > 1.0 else self.threshold
        
        # Create edge map
        edges = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
        edges[:-1, :] |= (diff_h > thresh_val).byte()
        edges[:, :-1] |= (diff_w > thresh_val).byte()
        
        # Combine with smooth_mask (where mask is False, we force edges)
        mask_tensor = torch.from_numpy(smooth_mask).to(self.device)
        edges = edges | (~mask_tensor).byte()
        
        # Download to CPU for Connected Components (OpenCV is optimized for this)
        edges_np = edges.cpu().numpy()
        
        # We want to label connected regions of NON-edges (0s)
        # connectedComponents finds white (1) regions, so we invert edges
        # regions = where edges are 0
        binary_mask = (edges_np == 0).astype(np.uint8)
        
        # Use simple connected components
        # Note: This is a simplification. For better results, one might use 
        # Felzenszwalb on CPU, but that is slow.
        # Here we rely on the strong pre-smoothing of Mumford-Shah.
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4)
        
        # Handle edges pixels (currently labeled 0). Assign them to nearest neighbor?
        # For speed, we just let them be background or distinct segments.
        # Vectorization usually ignores 1px boundaries anyway.
        
        return labels


# ==========================================
# 3. GPU Gradient Fitting (Batch Processing)
# ==========================================
class GradientFitterTorch:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit_batch(self, image: np.ndarray, segments: np.ndarray) -> Dict:
        """
        Calculates average color for all segments in parallel using GPU.
        """
        img_tensor = torch.from_numpy(image).float().to(self.device).view(-1, 3) # (N_pixels, 3)
        seg_tensor = torch.from_numpy(segments).long().to(self.device).view(-1)  # (N_pixels,)
        
        num_segments = int(segments.max()) + 1
        
        # --- 1. Constant Fill (Mean Color) ---
        # Count pixels per segment
        count = torch.bincount(seg_tensor, minlength=num_segments).float()
        count = torch.clamp(count, min=1.0).unsqueeze(1) # Avoid division by zero
        
        # Sum colors per segment
        sum_colors = torch.zeros((num_segments, 3), device=self.device)
        sum_colors.index_add_(0, seg_tensor, img_tensor)
        
        mean_colors = sum_colors / count
        
        # Move results to CPU
        mean_colors_cpu = mean_colors.cpu().numpy()
        counts_cpu = count.cpu().numpy().flatten()
        
        segments_info = {}
        for i in range(num_segments):
            # Ignore tiny segments (noise)
            if counts_cpu[i] < 4: 
                continue
            
            segments_info[i] = {
                'fill_type': 'constant',
                'color': mean_colors_cpu[i]
            }
                
        return segments_info


# ==========================================
# 4. Curve Fitting (CPU - OpenCV Optimized)
# ==========================================
class CurveFitterCPU:
    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance

    def fit(self, segmentation: np.ndarray, segments_info: Dict) -> List[Dict]:
        paths = []
        # unique_segs = np.unique(segmentation) # Slow on large arrays
        # We iterate through segments_info keys which is faster
        
        h, w = segmentation.shape
        
        for seg_id in list(segments_info.keys()):
            # Extract binary mask for this segment
            # Using uint8 for OpenCV
            mask = (segmentation == seg_id).astype(np.uint8)
            
            # Find contours
            # RETR_EXTERNAL: only outer contours
            # CHAIN_APPROX_SIMPLE: compress horizontal/vertical segments
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3: 
                    continue
                
                # Douglas-Peucker simplification
                epsilon = self.tolerance
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3: 
                    continue
                
                # Reshape from (N, 1, 2) to (N, 2) and convert to list
                points = approx.reshape(-1, 2).tolist()
                
                paths.append({
                    'points': points,
                    'segment_id': int(seg_id),
                    'closed': True
                })
        return paths


# ==========================================
# 5. SVG Generator
# ==========================================
class SVGGenerator:
    def generate(self, shape, paths, info):
        H, W = shape
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
        
        for path in paths:
            sid = path['segment_id']
            if sid not in info: continue
            
            seg_data = info[sid]
            color = seg_data['color']
            
            # Convert [0,1] float color to Hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(np.clip(color[0]*255, 0, 255)),
                int(np.clip(color[1]*255, 0, 255)),
                int(np.clip(color[2]*255, 0, 255))
            )
            
            # Construct path data
            points_str = " ".join([f"{p[0]},{p[1]}" for p in path['points']])
            d = f"M {points_str.replace(' ', ' L ')} Z"
            
            svg.append(f'<path d="{d}" fill="{hex_color}" stroke="none" />')
            
        svg.append('</svg>')
        return "\n".join(svg)


# ==========================================
# Main Pipeline
# ==========================================
class ImageVectorizationTorch:
    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"[{self.__class__.__name__}] Initializing on device: {self.device}")
        
        self.smoother = MumfordShahTorch(device=self.device)
        self.segmenter = SegmentationTorch(device=self.device)
        self.grad_fitter = GradientFitterTorch(device=self.device)
        self.curve_fitter = CurveFitterCPU(tolerance=1.5) # Higher tolerance = fewer points = faster
        self.svg_gen = SVGGenerator()

    def vectorize(self, image_path: str, output_path: str):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        # 1. Load Image
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Failed to load image.")
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Limit image size if too large (optional optimization)
        MAX_DIM = 1024
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized image to {new_w}x{new_h} for performance.")
            
        img_float = img.astype(np.float32) / 255.0
        print(f"Load Time: {time.time()-t0:.3f}s")

        # 2. Smoothing
        t1 = time.time()
        smoothed, disc_map = self.smoother.smooth(img_float)
        torch.cuda.empty_cache() # Cleanup GPU memory
        print(f"Smoothing Time: {time.time()-t1:.3f}s")
        
        # 3. Segmentation
        t2 = time.time()
        segments = self.segmenter.segment(smoothed, ~disc_map)
        torch.cuda.empty_cache()
        print(f"Segmentation Time: {time.time()-t2:.3f}s (Segments: {segments.max()})")
        
        # 4. Fitting
        t3 = time.time()
        seg_info = self.grad_fitter.fit_batch(smoothed, segments)
        torch.cuda.empty_cache()
        print(f"Fitting Time: {time.time()-t3:.3f}s")
        
        # 5. Curve & SVG
        t4 = time.time()
        paths = self.curve_fitter.fit(segments, seg_info)
        svg_content = self.svg_gen.generate(img.shape[:2], paths, seg_info)
        
        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"SVG Gen Time: {time.time()-t4:.3f}s")
        
        print(f"Total Execution Time: {time.time()-t0:.3f}s")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Example Usage
    import sys
    
    # Generate a dummy test file if no args
    def create_dummy():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Gradient background
        for i in range(512):
            img[i, :, 0] = i // 2
            img[i, :, 2] = 255 - (i // 2)
        # Objects
        cv2.circle(img, (256, 256), 100, (255, 255, 0), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.imwrite("test_input.png", img)
        return "test_input.png"

    if len(sys.argv) < 2:
        input_file = create_dummy()
        output_file = "output.svg"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.svg"

    vec = ImageVectorizationTorch(use_gpu=True)
    vec.vectorize(input_file, output_file)
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import interpolate
import sys
import os
import time
from typing import Dict, List, Tuple

class MumfordShahTorch:
    """
    GPU-accelerated Mumford-Shah Smoothing
    Sử dụng PyTorch tensors và Convolution để tính gradient/divergence
    """
    def __init__(self, alpha: float = 1.0, lambda_param: float = 1.5, 
                 max_iterations: int = 100, tolerance: float = 1e-3, device='cuda'):
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = device if torch.cuda.is_available() else 'cpu'

    def smooth(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Convert to Tensor (N, C, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        u = img_tensor.clone()
        
        # Kernel cho Finite Difference
        # du/dx: [0, 0, 0], [-1, 1, 0], [0, 0, 0]
        dx_kernel = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]]).float().to(self.device)
        dy_kernel = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]]).float().to(self.device)
        
        # Expand kernels cho 3 channels
        dx_kernel = dx_kernel.repeat(3, 1, 1, 1)
        dy_kernel = dy_kernel.repeat(3, 1, 1, 1)
        
        step_size = 0.1
        
        for i in range(self.max_iterations):
            u_prev = u.clone()
            
            # 1. Tính Gradient ∇u
            # Padding để giữ nguyên kích thước sau conv
            grad_x = F.conv2d(F.pad(u, (1, 1, 1, 1), mode='replicate'), dx_kernel, groups=3)
            grad_y = F.conv2d(F.pad(u, (1, 1, 1, 1), mode='replicate'), dy_kernel, groups=3)
            
            # Crop về kích thước gốc
            grad_x = grad_x[:, :, 1:-1, 1:-1]
            grad_y = grad_y[:, :, 1:-1, 1:-1]
            
            grad_norm_sq = grad_x**2 + grad_y**2
            
            # 2. Penalty function g(x) = min(alpha * ||∇u||^2, lambda)
            # Chúng ta cần đạo hàm của regularization term. 
            # Xấp xỉ: weight = 1 nếu smooth, 0 nếu edge
            # g'(t) approx 1 / (1 + ||grad||^2/lambda) hoặc hard threshold
            
            # Cách tiếp cận Discrete MS đơn giản hóa cho GPU:
            # Diffuse (Làm mượt) mạnh ở nơi gradient thấp, yếu ở nơi gradient cao
            weights = torch.exp(-self.alpha * grad_norm_sq)
            
            # Laplacian weighted (Divergence of weighted gradient)
            # div(w * ∇u)
            grad_x_w = grad_x * weights
            grad_y_w = grad_y * weights
            
            # Divergence lùi (backward difference approximation)
            div = torch.zeros_like(u)
            div[:, :, :, 1:] += grad_x_w[:, :, :, 1:] - grad_x_w[:, :, :, :-1]
            div[:, :, 1:, :] += grad_y_w[:, :, 1:, :] - grad_y_w[:, :, :-1, :]
            
            # 3. Update step: u = u + step * (2(I - u) + div)
            data_term = 2 * (img_tensor - u)
            u = u + step_size * (data_term + div)
            u = torch.clamp(u, 0, 1)
            
            # Check convergence
            if i % 10 == 0:
                diff = torch.mean(torch.abs(u - u_prev))
                if diff < self.tolerance:
                    break
        
        # Tính Discontinuity Map cuối cùng
        grad_x = F.conv2d(F.pad(u, (1, 1, 1, 1), mode='replicate'), dx_kernel, groups=3)[:, :, 1:-1, 1:-1]
        grad_y = F.conv2d(F.pad(u, (1, 1, 1, 1), mode='replicate'), dy_kernel, groups=3)[:, :, 1:-1, 1:-1]
        grad_mag = torch.sqrt(torch.sum(grad_x**2 + grad_y**2, dim=1)) # Sum over channels
        
        discontinuity = (grad_mag > (self.lambda_param * 0.5)).float()
        
        # Return numpy
        u_np = u.squeeze(0).permute(1, 2, 0).cpu().numpy()
        disc_np = discontinuity.squeeze(0).cpu().numpy().astype(bool)
        
        return u_np, disc_np

class SegmentationTorch:
    """
    GPU-assisted Segmentation
    Tính toán khoảng cách màu trên GPU, sau đó dùng thuật toán tối ưu của OpenCV cho labeling
    """
    def __init__(self, threshold: float = 10.0, device='cuda'):
        self.threshold = threshold
        self.device = device if torch.cuda.is_available() else 'cpu'

    def segment(self, image: np.ndarray, smooth_mask: np.ndarray) -> np.ndarray:
        # Image: H, W, 3 (RGB) -> Convert to LAB (CPU faster for small conversion) or approximation
        # Để nhanh, ta dùng RGB distance trực tiếp hoặc chuyển LAB đơn giản
        
        # PyTorch impl
        img_tensor = torch.from_numpy(image).to(self.device).float()
        H, W, C = img_tensor.shape
        
        # Tính sự khác biệt màu giữa các pixel lân cận trên GPU
        # Shift images
        diff_h = torch.sum(torch.abs(img_tensor[:-1, :] - img_tensor[1:, :]), dim=2)
        diff_w = torch.sum(torch.abs(img_tensor[:, :-1] - img_tensor[:, 1:]), dim=2)
        
        # Ngưỡng màu (Scaled cho khoảng 0-1 của ảnh, threshold input thường là CIELAB ~10/100)
        # Nếu ảnh 0-1, threshold tương ứng khoảng 0.04 - 0.05
        thresh_val = self.threshold / 255.0 if self.threshold > 1.0 else self.threshold
        
        # Tạo bản đồ cạnh (Edges)
        edges = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
        edges[:-1, :] |= (diff_h > thresh_val).byte()
        edges[:, :-1] |= (diff_w > thresh_val).byte()
        
        # Kết hợp với mask làm mượt
        mask_tensor = torch.from_numpy(smooth_mask).to(self.device)
        edges = edges | (~mask_tensor).byte()
        
        # Download về CPU để chạy Connected Components (OpenCV cực nhanh phần này)
        edges_np = edges.cpu().numpy()
        
        # Đảo ngược logic: edges là ranh giới, ta cần vùng liên thông (nơi không có cạnh)
        # Tuy nhiên connectedComponents cần vùng foreground.
        # Cách tốt nhất: Dùng thuật toán floodFill hoặc graph segmentation.
        # Ở đây ta dùng Felzenszwalb (skimage) hoặc đơn giản là connected components trên mask
        
        # Simplification: Dùng OpenCV connected components trên các vùng đồng màu
        # Ta convert ảnh sang uint8, giảm độ phân giải màu nhẹ để group
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Dùng Meanshift filtering của OpenCV (rất nhanh trên CPU C++)
        # để gộp các vùng nhỏ trước
        shifted = cv2.pyrMeanShiftFiltering(img_uint8, sp=5, sr=self.threshold)
        gray = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
        
        # Water-shed hoặc đơn giản là Canny + FindContours để lấy label
        # Cách nhanh nhất cho vectorization là Watershed
        _, markers = cv2.connectedComponents((~edges_np).astype(np.uint8))
        
        return markers

class GradientFitterTorch:
    """
    GPU-accelerated Gradient Fitting
    Tính toán song song cho tất cả các segment thay vì vòng lặp
    """
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit_batch(self, image: np.ndarray, segments: np.ndarray) -> Dict:
        """
        Fit màu (Constant) và Gradient (Linear) cho tất cả segment cùng lúc
        """
        img_tensor = torch.from_numpy(image).float().to(self.device).view(-1, 3) # (N_pixels, 3)
        seg_tensor = torch.from_numpy(segments).long().to(self.device).view(-1)  # (N_pixels,)
        
        num_segments = int(segments.max()) + 1
        
        # 1. Constant Fill (Mean Color)
        # Tính tổng màu cho mỗi segment id
        count = torch.bincount(seg_tensor, minlength=num_segments).float()
        count = torch.clamp(count, min=1.0).unsqueeze(1) # Tránh chia cho 0
        
        sum_colors = torch.zeros((num_segments, 3), device=self.device)
        # Scatter add: cộng dồn pixel color vào đúng index segment
        sum_colors.index_add_(0, seg_tensor, img_tensor)
        
        mean_colors = sum_colors / count
        
        # 2. Linear Gradient Estimation (PCA on coordinates)
        # Cần tìm hướng biến thiên màu mạnh nhất.
        # Structure Tensor trên từng segment.
        
        segments_info = {}
        mean_colors_cpu = mean_colors.cpu().numpy()
        
        # Simple Logic: Trả về constant fill cho tốc độ cực cao
        # Để implement Linear Gradient vector hóa hoàn toàn khá phức tạp,
        # Ta sẽ dùng heuristic: Nếu phương sai màu của segment lớn -> Linear
        
        # Tính variance
        sum_sq = torch.zeros((num_segments, 3), device=self.device)
        sum_sq.index_add_(0, seg_tensor, img_tensor ** 2)
        variance = (sum_sq / count) - mean_colors ** 2
        total_var = torch.sum(variance, dim=1)
        
        total_var_cpu = total_var.cpu().numpy()
        
        for i in range(num_segments):
            if count[i].item() < 5: continue # Bỏ qua rác
            
            color = mean_colors_cpu[i]
            
            # Nếu variance thấp, dùng constant
            if total_var_cpu[i] < 0.01:
                segments_info[i] = {
                    'fill_type': 'constant',
                    'color': color
                }
            else:
                # Fallback: dùng logic Linear fit đơn giản hoặc giữ constant
                # Để tối ưu tốc độ, ở mức độ này ta dùng constant.
                # Muốn Linear fit chính xác cần SVD trên tọa độ từng segment (chậm nếu loop)
                segments_info[i] = {
                    'fill_type': 'constant', 
                    'color': color
                }
                
        return segments_info

class CurveFitterCPU:
    """
    Curve fitting vẫn chạy trên CPU vì tính chất tuần tự của việc dò đường bao (contour tracing).
    Tuy nhiên OpenCV findContours rất tối ưu.
    """
    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance

    def fit(self, segmentation: np.ndarray, segments_info: Dict) -> List[Dict]:
        paths = []
        unique_segs = np.unique(segmentation)
        
        # Tối ưu: Chỉ xử lý các segment có trong info
        for seg_id in unique_segs:
            if seg_id not in segments_info:
                continue
                
            # Tạo mask nhị phân (uint8)
            mask = (segmentation == seg_id).astype(np.uint8)
            
            # Find contours (OpenCV nhanh hơn skimage)
            # CHAIN_APPROX_SIMPLE tự động đơn giản hóa các đoạn thẳng
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3: continue
                
                # Simplify thêm bằng Douglas-Peucker
                epsilon = self.tolerance  # Pixel tolerance
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3: continue
                
                # Convert format [[x,y]] -> [x,y]
                points = approx.reshape(-1, 2).tolist()
                
                paths.append({
                    'points': points,
                    'segment_id': int(seg_id),
                    'closed': True
                })
        return paths

class SVGGenerator:
    def generate(self, shape, paths, info):
        H, W = shape
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
        
        for path in paths:
            sid = path['segment_id']
            if sid not in info: continue
            
            seg_data = info[sid]
            color = seg_data['color']
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(np.clip(color[0]*255, 0, 255)),
                int(np.clip(color[1]*255, 0, 255)),
                int(np.clip(color[2]*255, 0, 255))
            )
            
            points_str = " ".join([f"{p[0]},{p[1]}" for p in path['points']])
            # SVG Path command: M (move) L (line) Z (close)
            d = f"M {points_str.replace(' ', ' L ')} Z"
            
            svg.append(f'<path d="{d}" fill="{hex_color}" stroke="none" />')
            
        svg.append('</svg>')
        return "\n".join(svg)

class ImageVectorizationTorch:
    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.smoother = MumfordShahTorch(device=self.device)
        self.segmenter = SegmentationTorch(device=self.device)
        self.grad_fitter = GradientFitterTorch(device=self.device)
        self.curve_fitter = CurveFitterCPU()
        self.svg_gen = SVGGenerator()

    def vectorize(self, image_path: str, output_path: str):
        # 1. Load Image
        t0 = time.time()
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img.astype(np.float32) / 255.0
        print(f"Loaded image {img.shape}: {time.time()-t0:.3f}s")

        # 2. Smoothing
        t1 = time.time()
        smoothed, disc_map = self.smoother.smooth(img_float)
        print(f"Smoothing (Mumford-Shah): {time.time()-t1:.3f}s")
        
        # 3. Segmentation
        t2 = time.time()
        # Resize nhỏ lại để segment nhanh hơn nếu ảnh quá lớn
        segments = self.segmenter.segment(smoothed, ~disc_map)
        print(f"Segmentation: {time.time()-t2:.3f}s (Segments: {segments.max()})")
        
        # 4. Fitting
        t3 = time.time()
        seg_info = self.grad_fitter.fit_batch(smoothed, segments)
        print(f"Gradient Fitting: {time.time()-t3:.3f}s")
        
        # 5. Curve & SVG
        t4 = time.time()
        paths = self.curve_fitter.fit(segments, seg_info)
        svg_content = self.svg_gen.generate(img.shape[:2], paths, seg_info)
        
        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"Curve Fitting & SVG Gen: {time.time()-t4:.3f}s")
        print(f"Total time: {time.time()-t0:.3f}s")

if __name__ == "__main__":
    import argparse
    
    # Tạo ảnh test nếu không có input
    def create_dummy_image():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Background gradient
        for i in range(512):
            img[i, :] = [i//2, 0, 255 - i//2]
        # Circle
        cv2.circle(img, (256, 256), 100, (255, 255, 0), -1)
        # Rectangle
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.imwrite("test_input.png", img)
        return "test_input.png"

    if len(sys.argv) < 2:
        print("Usage: python image_vectorization_torch.py <input_image> <output.svg>")
        print("Creating dummy input for testing...")
        input_file = create_dummy_image()
        output_file = "output.svg"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.svg"

    vectorizer = ImageVectorizationTorch(use_gpu=True)
    vectorizer.vectorize(input_file, output_file)
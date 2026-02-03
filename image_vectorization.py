"""
Image Vectorization via Gradient Reconstruction
Implementation based on the EUROGRAPHICS 2025 paper

Main vectorization pipeline following the paper's algorithm
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import min_cut
from skimage import color, measure
from skimage.morphology import dilation, square
import networkx as nx
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from mumford_shah import MumfordShahSmoother
from segmentation import DiscontSegmenter, ColorBasedSegmenter
from gradient_fitting import GradientFitter
from curve_fitting import CurveFitter
from svg_generator import SVGGenerator


class ImageVectorization:
    """
    Main class for image vectorization via gradient reconstruction
    
    Parameters from paper:
    - alpha: 1.0 (Mumford-Shah)
    - lambda_ms: 1.5 (Mumford-Shah)
    - tau_s: 10.0 (color difference threshold in CIELAB)
    - tau_a: 0.25 (discontinuity threshold)
    - sigma: 5 (discontinuity search distance)
    - lambda_grad: 1.0 (gradient stop fitting)
    - alpha_grad: infinity (enforce linear interpolation)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        lambda_ms: float = 1.5,
        tau_s: float = 10.0,
        tau_a: float = 0.25,
        sigma: int = 5,
        lambda_grad: float = 1.0,
        alpha_grad: float = np.inf
    ):
        self.alpha = alpha
        self.lambda_ms = lambda_ms
        self.tau_s = tau_s
        self.tau_a = tau_a
        self.sigma = sigma
        self.lambda_grad = lambda_grad
        self.alpha_grad = alpha_grad
        
        # Initialize components
        self.ms_smoother = MumfordShahSmoother(alpha=alpha, lambda_param=lambda_ms)
        self.color_segmenter = ColorBasedSegmenter(threshold=tau_s)
        self.discont_segmenter = DiscontSegmenter(tau_a=tau_a, sigma=sigma)
        self.gradient_fitter = GradientFitter(lambda_grad=lambda_grad, alpha_grad=alpha_grad)
        self.curve_fitter = CurveFitter()
        self.svg_generator = SVGGenerator()
        
    def vectorize(
        self, 
        image: np.ndarray,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Main vectorization pipeline
        
        Args:
            image: Input RGB image (H, W, 3) with values in [0, 255]
            verbose: Print progress information
            
        Returns:
            svg_content: SVG string
            metadata: Dictionary with intermediate results and statistics
        """
        if verbose:
            print("=" * 60)
            print("Image Vectorization via Gradient Reconstruction")
            print("=" * 60)
        
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        metadata = {
            'input_shape': image.shape,
            'steps': {}
        }
        
        # Step 1: Preprocessing (Mumford-Shah smoothing)
        if verbose:
            print("\n[Step 1/5] Preprocessing with Mumford-Shah smoothing...")
        
        smoothed, discontinuity = self.ms_smoother.smooth(image)
        metadata['steps']['preprocessing'] = {
            'smoothed_image': smoothed,
            'discontinuity_map': discontinuity,
            'discontinuity_pixels': int(discontinuity.sum())
        }
        
        if verbose:
            print(f"  - Discontinuity pixels: {discontinuity.sum()}")
        
        # Step 2: Discontinuity-Aware Segmentation
        if verbose:
            print("\n[Step 2/5] Discontinuity-aware segmentation...")
        
        # 2.1: Color-based segmentation on smooth regions
        smooth_mask = ~discontinuity
        S0 = self.color_segmenter.segment(smoothed, smooth_mask)
        
        if verbose:
            print(f"  - Initial segments (S0): {S0.max() + 1}")
        
        # 2.2: Graph-based multi-terminal min-cut
        S = self.discont_segmenter.segment(
            smoothed, 
            S0, 
            discontinuity,
            verbose=verbose
        )
        
        metadata['steps']['segmentation'] = {
            'S0_segments': int(S0.max() + 1),
            'S_segments': int(S.max() + 1),
            'segmentation_map': S
        }
        
        if verbose:
            print(f"  - Final smooth segments (S): {S.max() + 1}")
        
        # Step 3: Function Parameter Estimation
        if verbose:
            print("\n[Step 3/5] Function parameter estimation...")
        
        segments_info = self.gradient_fitter.fit_all_segments(
            image,
            smoothed,
            S,
            smooth_mask,
            verbose=verbose
        )
        
        metadata['steps']['gradient_fitting'] = segments_info
        
        # Count gradient types
        fill_types = [s['fill_type'] for s in segments_info.values()]
        if verbose:
            print(f"  - Constant fills: {fill_types.count('constant')}")
            print(f"  - Linear gradients: {fill_types.count('linear')}")
            print(f"  - Radial gradients: {fill_types.count('radial')}")
        
        # Step 4: Process discontinuity pixels
        if verbose:
            print("\n[Step 4/5] Processing discontinuity pixels...")
        
        S_final, segments_info = self._merge_discontinuity_pixels(
            image,
            smoothed,
            S,
            discontinuity,
            segments_info,
            verbose=verbose
        )
        
        metadata['steps']['final_segmentation'] = {
            'total_segments': int(S_final.max() + 1),
            'segmentation_map': S_final
        }
        
        if verbose:
            print(f"  - Total segments: {S_final.max() + 1}")
        
        # Step 5: Curve Fitting
        if verbose:
            print("\n[Step 5/5] Curve fitting and SVG generation...")
        
        paths = self.curve_fitter.fit_curves(S_final, segments_info)
        
        metadata['steps']['curve_fitting'] = {
            'num_paths': len(paths)
        }
        
        if verbose:
            print(f"  - Number of paths: {len(paths)}")
        
        # Generate SVG
        svg_content = self.svg_generator.generate(
            image.shape[:2],
            paths,
            segments_info
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Vectorization complete!")
            print("=" * 60)
        
        return svg_content, metadata
    
    def _merge_discontinuity_pixels(
        self,
        original_image: np.ndarray,
        smoothed: np.ndarray,
        S: np.ndarray,
        discontinuity: np.ndarray,
        segments_info: Dict,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Merge discontinuity pixels with neighboring segments
        
        Following Section 3.4 of the paper
        """
        S_final = S.copy()
        
        # Segment discontinuity pixels
        S_D = self.color_segmenter.segment(smoothed, discontinuity)
        
        # Offset discontinuity segment IDs to avoid conflicts
        max_S = S.max() + 1
        S_D_offset = S_D.copy()
        S_D_offset[discontinuity] += max_S
        
        # Merge
        S_final[discontinuity] = S_D_offset[discontinuity]
        
        # For each discontinuity segment, check if it should be merged
        unique_D_segs = np.unique(S_D_offset[discontinuity])
        
        merged_count = 0
        for d_seg in unique_D_segs:
            mask = (S_final == d_seg)
            
            # Find neighboring segments
            dilated = dilation(mask.astype(np.uint8), square(3))
            neighbors_mask = dilated & (~mask)
            neighbor_labels = S_final[neighbors_mask]
            neighbor_labels = neighbor_labels[neighbor_labels < max_S]  # Only original segments
            
            if len(neighbor_labels) == 0:
                # Add as constant fill
                mean_color = smoothed[mask].mean(axis=0)
                segments_info[int(d_seg)] = {
                    'fill_type': 'constant',
                    'color': mean_color,
                    'error': 0.0
                }
                continue
            
            neighbors = np.unique(neighbor_labels)
            
            # Compute color of discontinuity segment
            c_u = smoothed[mask].mean(axis=0)
            
            # Check if can be approximated by linear combination of neighbors
            can_merge = False
            best_neighbor = None
            min_dist = float('inf')
            
            for neighbor in neighbors:
                if neighbor not in segments_info:
                    continue
                    
                # Evaluate neighbor's fill function
                neighbor_info = segments_info[neighbor]
                
                # Simple check: distance to neighbor's color
                if neighbor_info['fill_type'] == 'constant':
                    neighbor_color = neighbor_info['color']
                    dist = np.linalg.norm(c_u - neighbor_color)
                    
                    if dist < self.tau_s:
                        can_merge = True
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = neighbor
            
            # Merge if appropriate
            if can_merge and best_neighbor is not None:
                S_final[mask] = best_neighbor
                merged_count += 1
            else:
                # Keep as separate constant fill
                segments_info[int(d_seg)] = {
                    'fill_type': 'constant',
                    'color': c_u,
                    'error': 0.0
                }
        
        if verbose:
            print(f"  - Merged {merged_count} discontinuity segments")
        
        return S_final, segments_info


def main():
    """Example usage"""
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description='Image Vectorization via Gradient Reconstruction')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output SVG path')
    parser.add_argument('--alpha', type=float, default=1.0, help='Mumford-Shah alpha')
    parser.add_argument('--lambda-ms', type=float, default=1.5, help='Mumford-Shah lambda')
    parser.add_argument('--tau-s', type=float, default=10.0, help='Color threshold')
    parser.add_argument('--tau-a', type=float, default=0.25, help='Discontinuity threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load image
    img = Image.open(args.input).convert('RGB')
    img_array = np.array(img)
    
    # Create vectorizer
    vectorizer = ImageVectorization(
        alpha=args.alpha,
        lambda_ms=args.lambda_ms,
        tau_s=args.tau_s,
        tau_a=args.tau_a
    )
    
    # Vectorize
    svg_content, metadata = vectorizer.vectorize(img_array, verbose=args.verbose)
    
    # Save SVG
    with open(args.output, 'w') as f:
        f.write(svg_content)
    
    print(f"\nSVG saved to: {args.output}")


if __name__ == '__main__':
    main()

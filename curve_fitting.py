"""
Curve Fitting Module
Converts segmentation to Bézier curves
Based on Section 3.5 of the paper
"""

import numpy as np
from skimage import measure
from scipy import interpolate
from typing import List, Dict, Tuple
import cv2


class CurveFitter:
    """
    Fit Bézier curves to segment boundaries
    
    Following Section 3.5:
    1. Trace boundaries as polylines
    2. Collect sequences into paths
    3. Connect paths at junctions
    4. Simplify using dynamic programming
    5. Combine into network of Bézier curves
    """
    
    def __init__(
        self,
        simplify_tolerance: float = 1.0,
        smoothness_weight: float = 0.5
    ):
        """
        Args:
            simplify_tolerance: Tolerance for curve simplification (pixels)
            smoothness_weight: Weight for smoothness in optimization
        """
        self.simplify_tolerance = simplify_tolerance
        self.smoothness_weight = smoothness_weight
    
    def fit_curves(
        self,
        segmentation: np.ndarray,
        segments_info: Dict
    ) -> List[Dict]:
        """
        Fit Bézier curves to all segment boundaries
        
        Args:
            segmentation: Segmentation map (H, W)
            segments_info: Dictionary with segment fill information
            
        Returns:
            List of path dictionaries, each containing:
            - 'points': List of (x, y) coordinates
            - 'segment_id': Segment ID
            - 'closed': Boolean indicating if path is closed
        """
        paths = []
        
        # Process each segment
        unique_segments = np.unique(segmentation[segmentation >= 0])
        
        for seg_id in unique_segments:
            if seg_id not in segments_info:
                continue
            
            # Create binary mask
            mask = (segmentation == seg_id).astype(np.uint8)
            
            # Find contours
            contours = measure.find_contours(mask, 0.5)
            
            for contour in contours:
                # Contour is (N, 2) array of [row, col] = [y, x]
                if len(contour) < 3:
                    continue
                
                # Convert to [x, y] format
                points = contour[:, [1, 0]]
                
                # Simplify contour
                simplified = self._simplify_polyline(points)
                
                # Fit Bézier curve (or keep as polyline for now)
                # For simplicity, we'll use the simplified polyline
                # A full implementation would fit cubic Bézier curves
                
                path = {
                    'points': simplified.tolist(),
                    'segment_id': int(seg_id),
                    'closed': True  # Contours are closed
                }
                
                paths.append(path)
        
        return paths
    
    def _simplify_polyline(
        self,
        points: np.ndarray,
        epsilon: float = None
    ) -> np.ndarray:
        """
        Simplify polyline using Douglas-Peucker algorithm
        
        Args:
            points: Array of points (N, 2)
            epsilon: Tolerance (defaults to self.simplify_tolerance)
            
        Returns:
            Simplified points array
        """
        if epsilon is None:
            epsilon = self.simplify_tolerance
        
        # Use OpenCV's approxPolyDP
        points_int = points.astype(np.float32)
        
        # Check if closed
        is_closed = np.allclose(points[0], points[-1])
        
        simplified = cv2.approxPolyDP(
            points_int,
            epsilon,
            closed=is_closed
        )
        
        # Remove extra dimension added by OpenCV
        if simplified.shape[1] == 1:
            simplified = simplified[:, 0, :]
        
        return simplified
    
    def _fit_bezier_curve(
        self,
        points: np.ndarray,
        num_control_points: int = None
    ) -> np.ndarray:
        """
        Fit cubic Bézier curve to points
        
        This is a simplified version. Full implementation would use
        the method from [BLP10] as mentioned in the paper.
        
        Args:
            points: Array of points (N, 2)
            num_control_points: Number of control points
            
        Returns:
            Control points for Bézier curve(s)
        """
        N = len(points)
        
        if num_control_points is None:
            # Estimate based on complexity
            num_control_points = max(4, N // 10)
        
        # Use spline interpolation as approximation
        # For a proper implementation, would fit cubic Bézier segments
        
        # Parameterize points by arc length
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        distances = np.concatenate([[0], np.cumsum(distances)])
        distances = distances / distances[-1]  # Normalize to [0, 1]
        
        # Fit spline
        tck, u = interpolate.splprep([points[:, 0], points[:, 1]], s=0, k=3)
        
        # Sample control points
        u_new = np.linspace(0, 1, num_control_points)
        control_points = np.array(interpolate.splev(u_new, tck)).T
        
        return control_points
    
    def _connect_paths_at_junctions(
        self,
        paths: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Connect paths at junction nodes for smoothness
        
        This is mentioned in step 3 of Section 3.5
        """
        # Simplified implementation
        # Full version would analyze junction nodes and maximize continuity
        return paths
    
    def _dynamic_programming_simplification(
        self,
        path: np.ndarray,
        junctions: List[int]
    ) -> np.ndarray:
        """
        Simplify path using dynamic programming
        
        Keep junction nodes fixed (step 4 of Section 3.5)
        """
        # Simplified implementation
        return self._simplify_polyline(path)


class BezierCurve:
    """Helper class for Bézier curve operations"""
    
    @staticmethod
    def cubic_bezier(t: float, p0: np.ndarray, p1: np.ndarray, 
                     p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """
        Evaluate cubic Bézier curve at parameter t
        
        B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        """
        return (
            (1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3
        )
    
    @staticmethod
    def fit_cubic_bezier(
        points: np.ndarray,
        t_values: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a single cubic Bézier curve to points
        
        Returns:
            p0, p1, p2, p3: Control points
        """
        N = len(points)
        
        if t_values is None:
            # Parameterize by arc length
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            distances = np.concatenate([[0], np.cumsum(distances)])
            t_values = distances / distances[-1]
        
        # End points are fixed
        p0 = points[0]
        p3 = points[-1]
        
        # Solve for interior control points using least squares
        # This is a simplified approach
        # Full implementation would use more sophisticated fitting
        
        # For now, use simple approximation
        p1 = points[0] + (points[-1] - points[0]) / 3
        p2 = points[0] + 2 * (points[-1] - points[0]) / 3
        
        return p0, p1, p2, p3
    
    @staticmethod
    def to_svg_path(control_points: List[np.ndarray], closed: bool = False) -> str:
        """
        Convert control points to SVG path string
        
        Args:
            control_points: List of (x, y) points
            closed: Whether path should be closed
            
        Returns:
            SVG path data string
        """
        if len(control_points) == 0:
            return ""
        
        path_parts = []
        
        # Move to first point
        path_parts.append(f"M {control_points[0][0]:.2f},{control_points[0][1]:.2f}")
        
        # Line segments
        for point in control_points[1:]:
            path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        # Close path if needed
        if closed:
            path_parts.append("Z")
        
        return " ".join(path_parts)


def test_curve_fitting():
    """Test curve fitting"""
    import matplotlib.pyplot as plt
    
    # Create simple segmentation
    seg = np.zeros((100, 100), dtype=np.int32)
    seg[20:80, 20:80] = 1
    seg[40:60, 40:60] = 2
    
    # Create dummy segments info
    segments_info = {
        0: {'fill_type': 'constant', 'color': [1, 1, 1]},
        1: {'fill_type': 'constant', 'color': [1, 0, 0]},
        2: {'fill_type': 'constant', 'color': [0, 1, 0]}
    }
    
    # Fit curves
    fitter = CurveFitter()
    paths = fitter.fit_curves(seg, segments_info)
    
    print(f"Number of paths: {len(paths)}")
    
    # Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(seg, cmap='tab10')
    
    for path in paths:
        points = np.array(path['points'])
        if len(points) > 0:
            plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
            plt.plot(points[:, 0], points[:, 1], 'ro', markersize=3)
    
    plt.title('Fitted Curves')
    plt.axis('equal')
    plt.savefig('/home/claude/curve_fitting_test.png', dpi=150)
    print("Test saved to curve_fitting_test.png")


if __name__ == '__main__':
    test_curve_fitting()

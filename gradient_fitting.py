"""
Gradient Fitting Module
Implements constant, linear, and radial gradient fitting
Based on Section 3.3 of the paper
"""

import numpy as np
from scipy.optimize import minimize
from scipy import ndimage
from skimage import color
from typing import Dict, Tuple, List


class GradientFitter:
    """
    Fit constant, linear, and radial gradient functions to image segments
    """
    
    def __init__(
        self,
        lambda_grad: float = 1.0,
        alpha_grad: float = np.inf,
        error_threshold: float = 15.0
    ):
        """
        Args:
            lambda_grad: Lambda parameter for gradient stop fitting
            alpha_grad: Alpha parameter (use inf for linear interpolation)
            error_threshold: Maximum acceptable L1 error in CIELAB
        """
        self.lambda_grad = lambda_grad
        self.alpha_grad = alpha_grad
        self.error_threshold = error_threshold
    
    def fit_constant(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Fit constant color function (K0)
        
        Uses weighted mean where weights decrease near boundaries
        
        Returns:
            color: RGB color (3,)
            error: L1 error in CIELAB
        """
        if not np.any(mask):
            return np.array([0.5, 0.5, 0.5]), 0.0
        
        # Compute distance from boundary
        from scipy.ndimage import distance_transform_edt
        boundary_dist = distance_transform_edt(mask)
        
        # Weights: higher for pixels farther from boundary
        weights = boundary_dist + 1.0
        weights = weights / weights.sum()
        
        # Weighted mean color
        color_rgb = np.zeros(3)
        for c in range(3):
            color_rgb[c] = np.sum(image[:, :, c] * weights * mask) / np.sum(weights * mask)
        
        # Compute error
        reconstruction = np.zeros_like(image)
        reconstruction[mask] = color_rgb
        
        error = self._compute_error(image, reconstruction, mask)
        
        return color_rgb, error
    
    def fit_linear(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[Dict, float]:
        """
        Fit linear gradient function (K1)
        
        Following Section 3.3.2:
        1. Find gradient direction d using structure tensor
        2. Compute color profile ρ(x) along direction d
        3. Fit piecewise linear approximation using 1D Mumford-Shah
        
        Returns:
            params: Dictionary with 'direction', 'stops' (list of (position, color))
            error: L1 error in CIELAB
        """
        if not np.any(mask):
            return {'direction': np.array([1, 0]), 'stops': []}, float('inf')
        
        H, W, C = image.shape
        
        # Compute image gradient
        gradient = np.zeros((H, W, 2, C))
        for c in range(C):
            gy, gx = np.gradient(image[:, :, c])
            gradient[:, :, 0, c] = gx
            gradient[:, :, 1, c] = gy
        
        # Structure tensor: T(p) = ∇I(p) · ∇I(p)^T
        # Summed over channels and pixels in mask
        structure_tensor = np.zeros((2, 2))
        
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                
                # Gradient vector for this pixel (averaged over channels)
                grad_vec = gradient[i, j].mean(axis=1)  # (2,)
                
                # Outer product
                structure_tensor += np.outer(grad_vec, grad_vec)
        
        # Find principal direction (eigenvector with largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(structure_tensor)
        direction = eigenvectors[:, -1]  # Last eigenvector (largest eigenvalue)
        
        # Normalize
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1.0, 0.0])
        
        # Compute color profile ρ(x) along direction d
        # ρ(x) = mean{I(p) | p ∈ s and d^T p = x}
        
        # Get pixel coordinates
        coords = np.argwhere(mask)  # (N, 2) - [y, x] format
        
        if len(coords) == 0:
            return {'direction': direction, 'stops': []}, float('inf')
        
        # Project onto direction (using x, y order for direction)
        # Direction is [dx, dy], coords are [y, x]
        projections = coords[:, 1] * direction[0] + coords[:, 0] * direction[1]
        
        # Get colors
        colors = image[coords[:, 0], coords[:, 1]]  # (N, 3)
        
        # Sort by projection
        sort_idx = np.argsort(projections)
        projections_sorted = projections[sort_idx]
        colors_sorted = colors[sort_idx]
        
        # Bin into discrete positions and average colors
        num_bins = min(len(projections_sorted), 100)
        bins = np.linspace(projections_sorted.min(), projections_sorted.max(), num_bins)
        
        bin_colors = []
        bin_positions = []
        
        for i in range(len(bins) - 1):
            bin_mask = (projections_sorted >= bins[i]) & (projections_sorted < bins[i+1])
            if np.any(bin_mask):
                bin_colors.append(colors_sorted[bin_mask].mean(axis=0))
                bin_positions.append((bins[i] + bins[i+1]) / 2)
        
        if len(bin_colors) < 2:
            # Fall back to constant
            color, _ = self.fit_constant(image, mask)
            return {
                'direction': direction,
                'stops': [(0, color)]
            }, float('inf')
        
        bin_colors = np.array(bin_colors)
        bin_positions = np.array(bin_positions)
        
        # Fit gradient stops using 1D Mumford-Shah on gradient of ρ
        stops = self._fit_gradient_stops(bin_positions, bin_colors)
        
        # Compute reconstruction error
        reconstruction = self._reconstruct_linear(
            image.shape[:2],
            mask,
            direction,
            stops
        )
        
        error = self._compute_error(image, reconstruction, mask)
        
        params = {
            'direction': direction,
            'stops': stops
        }
        
        return params, error
    
    def fit_radial(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[Dict, float]:
        """
        Fit radial gradient function (K2)
        
        Following Section 3.3.3:
        1. Find focal point φ and eccentricity e by minimizing gradient misalignment
        2. Compute radial color profile ρ(r)
        3. Fit gradient stops
        
        Returns:
            params: Dictionary with 'focus', 'eccentricity', 'center', 'transform', 'stops'
            error: L1 error in CIELAB
        """
        if not np.any(mask):
            return {
                'focus': np.array([0, 0]),
                'eccentricity': np.array([0, 0]),
                'stops': []
            }, float('inf')
        
        H, W, C = image.shape
        
        # Compute image gradient
        gradient = np.zeros((H, W, 2, C))
        for c in range(C):
            gy, gx = np.gradient(image[:, :, c])
            gradient[:, :, 0, c] = gx
            gradient[:, :, 1, c] = gy
        
        # Get pixels in segment
        coords = np.argwhere(mask)  # (N, 2) in [y, x] format
        
        if len(coords) < 10:
            # Not enough pixels for radial fit
            color, _ = self.fit_constant(image, mask)
            return {
                'focus': np.array([W/2, H/2]),
                'eccentricity': np.array([0, 0]),
                'stops': [(0, color)]
            }, float('inf')
        
        # Initial guess: center of mass
        center_y = coords[:, 0].mean()
        center_x = coords[:, 1].mean()
        
        # Optimize focus and eccentricity
        # Using simplified approach: focus at center, zero eccentricity
        focus = np.array([center_x, center_y])
        eccentricity = np.array([0.0, 0.0])
        
        # Compute radial distances
        radii = []
        colors = []
        
        for coord in coords:
            y, x = coord
            r = np.sqrt((x - focus[0])**2 + (y - focus[1])**2)
            radii.append(r)
            colors.append(image[y, x])
        
        radii = np.array(radii)
        colors = np.array(colors)
        
        # Sort by radius
        sort_idx = np.argsort(radii)
        radii_sorted = radii[sort_idx]
        colors_sorted = colors[sort_idx]
        
        # Bin radii and average colors
        num_bins = min(len(radii_sorted), 50)
        if num_bins < 2:
            color, _ = self.fit_constant(image, mask)
            return {
                'focus': focus,
                'eccentricity': eccentricity,
                'stops': [(0, color)]
            }, float('inf')
        
        bins = np.linspace(radii_sorted.min(), radii_sorted.max(), num_bins)
        
        bin_colors = []
        bin_radii = []
        
        for i in range(len(bins) - 1):
            bin_mask = (radii_sorted >= bins[i]) & (radii_sorted < bins[i+1])
            if np.any(bin_mask):
                bin_colors.append(colors_sorted[bin_mask].mean(axis=0))
                bin_radii.append((bins[i] + bins[i+1]) / 2)
        
        if len(bin_colors) < 2:
            color, _ = self.fit_constant(image, mask)
            return {
                'focus': focus,
                'eccentricity': eccentricity,
                'stops': [(0, color)]
            }, float('inf')
        
        bin_colors = np.array(bin_colors)
        bin_radii = np.array(bin_radii)
        
        # Fit gradient stops
        stops = self._fit_gradient_stops(bin_radii, bin_colors)
        
        # Compute reconstruction error
        reconstruction = self._reconstruct_radial(
            image.shape[:2],
            mask,
            focus,
            eccentricity,
            stops
        )
        
        error = self._compute_error(image, reconstruction, mask)
        
        params = {
            'focus': focus,
            'eccentricity': eccentricity,
            'center': np.array([0, 0]),  # Origin
            'transform': np.eye(2),  # Identity
            'stops': stops
        }
        
        return params, error
    
    def _fit_gradient_stops(
        self,
        positions: np.ndarray,
        colors: np.ndarray
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Fit gradient stops using 1D Mumford-Shah on gradient of ρ
        
        Equation (5) from paper
        """
        if len(positions) < 2:
            return [(positions[0], colors[0])]
        
        # Compute gradient of color profile
        grad_colors = np.diff(colors, axis=0)
        
        # Simple piecewise constant approximation
        # Find discontinuities (large color changes)
        grad_magnitude = np.linalg.norm(grad_colors, axis=1)
        
        # Threshold for detecting stops
        threshold = self.lambda_grad
        discontinuities = grad_magnitude > threshold
        
        # Start and end are always stops
        stop_indices = [0]
        
        for i in range(len(discontinuities)):
            if discontinuities[i]:
                stop_indices.append(i + 1)
        
        stop_indices.append(len(positions) - 1)
        
        # Remove duplicates and sort
        stop_indices = sorted(set(stop_indices))
        
        # Create stops
        stops = []
        for idx in stop_indices:
            stops.append((positions[idx], colors[idx]))
        
        return stops
    
    def _reconstruct_linear(
        self,
        shape: Tuple[int, int],
        mask: np.ndarray,
        direction: np.ndarray,
        stops: List[Tuple[float, np.ndarray]]
    ) -> np.ndarray:
        """Reconstruct image using linear gradient"""
        H, W = shape
        reconstruction = np.zeros((H, W, 3))
        
        if len(stops) == 0:
            return reconstruction
        
        # Create coordinate grid
        y, x = np.mgrid[0:H, 0:W]
        
        # Project coordinates onto gradient direction
        projections = x * direction[0] + y * direction[1]
        
        # Interpolate colors
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                
                proj = projections[i, j]
                
                # Find surrounding stops
                color = self._interpolate_gradient(proj, stops)
                reconstruction[i, j] = color
        
        return reconstruction
    
    def _reconstruct_radial(
        self,
        shape: Tuple[int, int],
        mask: np.ndarray,
        focus: np.ndarray,
        eccentricity: np.ndarray,
        stops: List[Tuple[float, np.ndarray]]
    ) -> np.ndarray:
        """Reconstruct image using radial gradient"""
        H, W = shape
        reconstruction = np.zeros((H, W, 3))
        
        if len(stops) == 0:
            return reconstruction
        
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                
                # Compute radius
                r = np.sqrt((j - focus[0])**2 + (i - focus[1])**2)
                
                # Interpolate color
                color = self._interpolate_gradient(r, stops)
                reconstruction[i, j] = color
        
        return reconstruction
    
    def _interpolate_gradient(
        self,
        position: float,
        stops: List[Tuple[float, np.ndarray]]
    ) -> np.ndarray:
        """Linear interpolation between gradient stops"""
        if len(stops) == 0:
            return np.array([0.5, 0.5, 0.5])
        
        if len(stops) == 1:
            return stops[0][1]
        
        # Find surrounding stops
        for i in range(len(stops) - 1):
            pos1, color1 = stops[i]
            pos2, color2 = stops[i + 1]
            
            if pos1 <= position <= pos2:
                # Linear interpolation
                if pos2 - pos1 > 0:
                    t = (position - pos1) / (pos2 - pos1)
                else:
                    t = 0
                return (1 - t) * color1 + t * color2
        
        # Outside range: use nearest
        if position < stops[0][0]:
            return stops[0][1]
        else:
            return stops[-1][1]
    
    def _compute_error(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Compute L1 error in CIELAB color space"""
        if not np.any(mask):
            return 0.0
        
        # Convert to LAB
        original_lab = color.rgb2lab(original)
        reconstruction_lab = color.rgb2lab(reconstruction)
        
        # Compute L1 error
        diff = np.abs(original_lab - reconstruction_lab)
        error = diff[mask].sum() / mask.sum()
        
        return error
    
    def fit_all_segments(
        self,
        original_image: np.ndarray,
        smoothed_image: np.ndarray,
        segmentation: np.ndarray,
        smooth_mask: np.ndarray,
        verbose: bool = False
    ) -> Dict[int, Dict]:
        """
        Fit gradient functions to all segments
        
        Returns:
            Dictionary mapping segment ID to fit parameters
        """
        segments = np.unique(segmentation[smooth_mask])
        segments_info = {}
        
        for seg_id in segments:
            mask = (segmentation == seg_id) & smooth_mask
            
            if not np.any(mask):
                continue
            
            # Try all three function types
            const_params, const_error = self.fit_constant(smoothed_image, mask)
            linear_params, linear_error = self.fit_linear(smoothed_image, mask)
            radial_params, radial_error = self.fit_radial(smoothed_image, mask)
            
            # Choose best fit
            errors = {
                'constant': const_error,
                'linear': linear_error,
                'radial': radial_error
            }
            
            best_type = min(errors, key=errors.get)
            best_error = errors[best_type]
            
            # If error too high, fall back to S0 segmentation
            if best_error > self.error_threshold:
                # Mark for re-segmentation
                segments_info[int(seg_id)] = {
                    'fill_type': 'needs_resegment',
                    'error': best_error
                }
            else:
                if best_type == 'constant':
                    segments_info[int(seg_id)] = {
                        'fill_type': 'constant',
                        'color': const_params,
                        'error': const_error
                    }
                elif best_type == 'linear':
                    segments_info[int(seg_id)] = {
                        'fill_type': 'linear',
                        'direction': linear_params['direction'],
                        'stops': linear_params['stops'],
                        'error': linear_error
                    }
                else:  # radial
                    segments_info[int(seg_id)] = {
                        'fill_type': 'radial',
                        'focus': radial_params['focus'],
                        'eccentricity': radial_params['eccentricity'],
                        'stops': radial_params['stops'],
                        'error': radial_error
                    }
        
        return segments_info

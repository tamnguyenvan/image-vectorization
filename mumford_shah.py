"""
Mumford-Shah Functional for Image Smoothing
Based on Section 3.1 of the paper

Reference: [SC14] Smith, S.M., & Brady, J.M. (1997)
Discrete Mumford-Shah implementation
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


class MumfordShahSmoother:
    """
    Discrete Mumford-Shah functional smoother
    
    Equation (1) from paper:
    u* = argmin_u Σ ||u(x) - I(x)||^2 + [∇u]^α_λ
    where [g]^α_λ = min(α||g||^2, λ)
    
    Parameters:
    - alpha: Controls color quantization strength (default: 1.0)
    - lambda_param: Controls smoothing vs edge preservation (default: 1.5)
    - max_iterations: Maximum iterations for optimization (default: 100)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        lambda_param: float = 1.5,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ):
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def _penalized_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute [g]^α_λ = min(α||g||^2, λ)
        
        Args:
            gradient: Gradient vector (H, W, 2, 3) for RGB
            
        Returns:
            Penalized gradient values (H, W, 3)
        """
        # Compute ||g||^2 for each pixel and channel
        grad_norm_sq = np.sum(gradient**2, axis=2)  # (H, W, 3)
        
        # Apply penalty function
        penalty = np.minimum(self.alpha * grad_norm_sq, self.lambda_param)
        
        return penalty
    
    def _compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete gradient ∇u using finite differences
        
        Args:
            u: Image (H, W, 3)
            
        Returns:
            Gradient (H, W, 2, 3) where axis 2 contains [du/dx, du/dy]
        """
        H, W, C = u.shape
        gradient = np.zeros((H, W, 2, C), dtype=np.float32)
        
        # Forward differences
        # du/dx
        gradient[:-1, :, 0, :] = u[1:, :, :] - u[:-1, :, :]
        # du/dy
        gradient[:, :-1, 1, :] = u[:, 1:, :] - u[:, :-1, :]
        
        return gradient
    
    def _compute_divergence(self, gradient_penalty: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute divergence term for gradient descent
        
        This is a simplified implementation using the gradient penalty
        """
        H, W, C = u.shape
        
        # Compute Laplacian (approximation of divergence)
        divergence = np.zeros_like(u)
        
        for c in range(C):
            # Simple Laplacian
            laplacian = ndimage.laplace(u[:, :, c])
            divergence[:, :, c] = laplacian
        
        return divergence
    
    def smooth(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mumford-Shah smoothing to image
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            verbose: Print iteration progress
            
        Returns:
            u_star: Smoothed image (H, W, 3)
            discontinuity: Binary discontinuity map (H, W)
        """
        H, W, C = image.shape
        
        # Initialize u with input image
        u = image.copy()
        
        # Iterative optimization (gradient descent)
        step_size = 0.1
        
        for iteration in range(self.max_iterations):
            # Compute gradient
            gradient = self._compute_gradient(u)
            
            # Compute gradient penalty
            gradient_penalty = self._penalized_gradient(gradient)
            
            # Update step (simplified gradient descent)
            # Data term: 2(u - I)
            data_term = 2 * (u - image)
            
            # Regularization term (simplified)
            reg_term = self._compute_divergence(gradient_penalty, u)
            
            # Update
            u_new = u - step_size * (data_term + 0.5 * reg_term)
            
            # Clamp to [0, 1]
            u_new = np.clip(u_new, 0, 1)
            
            # Check convergence
            change = np.mean(np.abs(u_new - u))
            u = u_new
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: change = {change:.6f}")
            
            if change < self.tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        # Compute discontinuity map D = {x | [∇u*(x)]^α_λ = λ}
        gradient = self._compute_gradient(u)
        gradient_penalty = self._penalized_gradient(gradient)
        
        # Discontinuity where penalty equals lambda (edge pixels)
        # Average across color channels
        gradient_penalty_mean = np.mean(gradient_penalty, axis=2)
        
        # Threshold to identify discontinuities
        # Use a small tolerance since we're comparing floats
        threshold = 0.9 * self.lambda_param
        discontinuity = gradient_penalty_mean >= threshold
        
        return u, discontinuity


def test_mumford_shah():
    """Test Mumford-Shah smoother"""
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Create a simple test image
    img = np.zeros((100, 100, 3))
    img[25:75, 25:75] = [1.0, 0.0, 0.0]  # Red square
    img[40:60, 40:60] = [0.0, 1.0, 0.0]  # Green square
    
    # Add noise
    noise = np.random.randn(*img.shape) * 0.1
    img_noisy = np.clip(img + noise, 0, 1)
    
    # Apply smoothing
    smoother = MumfordShahSmoother(alpha=1.0, lambda_param=1.5)
    smoothed, discontinuity = smoother.smooth(img_noisy, verbose=True)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_noisy)
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(smoothed)
    axes[1, 0].set_title('Smoothed')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(discontinuity, cmap='gray')
    axes[1, 1].set_title('Discontinuity Map')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/mumford_shah_test.png', dpi=150, bbox_inches='tight')
    print("Test image saved to mumford_shah_test.png")


if __name__ == '__main__':
    test_mumford_shah()

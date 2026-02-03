"""
Segmentation modules for image vectorization
Implements color-based segmentation and discontinuity-aware segmentation
Based on Section 3.2 of the paper
"""

import numpy as np
from scipy import ndimage
from skimage import color, measure
from skimage.morphology import dilation, square
import networkx as nx
from typing import Tuple, Set, Dict, List
from collections import defaultdict


class ColorBasedSegmenter:
    """
    Color-difference based segmentation
    Segments pixels based on CIELAB color distance
    """
    
    def __init__(self, threshold: float = 10.0):
        """
        Args:
            threshold: Color difference threshold in CIELAB (tau_s in paper)
        """
        self.threshold = threshold
    
    def segment(
        self,
        image: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Segment image based on color differences
        
        Args:
            image: RGB image (H, W, 3) with values in [0, 1]
            mask: Boolean mask indicating which pixels to segment
            
        Returns:
            Segmentation map (H, W) with integer labels
        """
        H, W, C = image.shape
        
        # Convert to CIELAB for perceptual color distance
        lab_image = color.rgb2lab(image)
        
        if mask is None:
            mask = np.ones((H, W), dtype=bool)
        
        # Initialize: each pixel is its own segment
        labels = np.arange(H * W).reshape(H, W)
        labels[~mask] = -1  # Mark non-masked pixels
        
        # Union-Find structure for efficient merging
        parent = np.arange(H * W)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Merge neighboring pixels with similar colors
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                
                current_idx = i * W + j
                current_color = lab_image[i, j]
                
                # Check 4-connected neighbors
                neighbors = []
                if i > 0 and mask[i-1, j]:
                    neighbors.append((i-1, j))
                if j > 0 and mask[i, j-1]:
                    neighbors.append((i, j-1))
                
                for ni, nj in neighbors:
                    neighbor_idx = ni * W + nj
                    neighbor_color = lab_image[ni, nj]
                    
                    # Compute CIELAB distance
                    color_dist = np.linalg.norm(current_color - neighbor_color)
                    
                    if color_dist < self.threshold:
                        union(current_idx, neighbor_idx)
        
        # Assign final labels
        label_map = {}
        next_label = 0
        
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    labels[i, j] = -1
                    continue
                
                idx = i * W + j
                root = find(idx)
                
                if root not in label_map:
                    label_map[root] = next_label
                    next_label += 1
                
                labels[i, j] = label_map[root]
        
        return labels


class DiscontSegmenter:
    """
    Discontinuity-aware segmentation using graph-based multi-terminal min-cut
    Based on Section 3.2 of the paper
    """
    
    def __init__(
        self,
        tau_a: float = 0.25,
        sigma: int = 5
    ):
        """
        Args:
            tau_a: Discontinuity threshold for A relation
            sigma: Search distance for discontinuity pairs (pixels)
        """
        self.tau_a = tau_a
        self.sigma = sigma
    
    def _build_segment_graph(
        self,
        image: np.ndarray,
        S0: np.ndarray
    ) -> Tuple[nx.Graph, Dict]:
        """
        Build weighted graph G(V, E, W) from segmentation S0
        
        Returns:
            graph: NetworkX graph
            segment_info: Dictionary with segment information
        """
        graph = nx.Graph()
        
        # Get unique segments
        segments = np.unique(S0[S0 >= 0])
        
        # Compute mean color for each segment
        segment_colors = {}
        segment_pixels = {}
        
        for seg in segments:
            mask = (S0 == seg)
            segment_colors[seg] = image[mask].mean(axis=0)
            segment_pixels[seg] = mask
        
        # Add nodes
        for seg in segments:
            graph.add_node(seg)
        
        # Add edges between neighboring segments
        H, W = S0.shape
        
        for i in range(H):
            for j in range(W):
                if S0[i, j] < 0:
                    continue
                
                current_seg = S0[i, j]
                
                # Check right and down neighbors
                neighbors = []
                if j < W - 1 and S0[i, j+1] >= 0:
                    neighbors.append(S0[i, j+1])
                if i < H - 1 and S0[i+1, j] >= 0:
                    neighbors.append(S0[i+1, j])
                
                for neighbor_seg in neighbors:
                    if neighbor_seg != current_seg:
                        # Compute edge weight: exp(-||I(u) - I(v)||^2)
                        color_diff = segment_colors[current_seg] - segment_colors[neighbor_seg]
                        weight = np.exp(-np.sum(color_diff**2))
                        
                        if not graph.has_edge(current_seg, neighbor_seg):
                            graph.add_edge(current_seg, neighbor_seg, weight=weight)
        
        segment_info = {
            'colors': segment_colors,
            'pixels': segment_pixels
        }
        
        return graph, segment_info
    
    def _compute_discontinuity_relation(
        self,
        S0: np.ndarray,
        discontinuity: np.ndarray,
        segment_info: Dict
    ) -> Set[Tuple[int, int]]:
        """
        Compute discontinuity relation A from Equation (3)
        
        A = {(u,v) | f_A(u,v) > tau_a * ||∂(u) ∩ ∂(v)||_0}
        """
        H, W = S0.shape
        
        # Count discontinuity pixels between segment pairs
        f_A = defaultdict(int)
        boundary_counts = defaultdict(int)
        
        # Find discontinuity pixels
        discont_pixels = np.argwhere(discontinuity)
        
        # Directions to check
        directions = {
            'right': (0, 1),
            'down': (1, 0),
            'diag': (1, 1)
        }
        
        for dy, dx in discont_pixels:
            # For each direction
            for dir_name, (di, dj) in directions.items():
                # Look in positive and negative directions
                u_seg = None
                v_seg = None
                
                # Positive direction
                for dist in range(1, self.sigma + 1):
                    ny, nx = dy + di * dist, dx + dj * dist
                    if 0 <= ny < H and 0 <= nx < W and S0[ny, nx] >= 0:
                        u_seg = S0[ny, nx]
                        break
                
                # Negative direction
                for dist in range(1, self.sigma + 1):
                    ny, nx = dy - di * dist, dx - dj * dist
                    if 0 <= ny < H and 0 <= nx < W and S0[ny, nx] >= 0:
                        v_seg = S0[ny, nx]
                        break
                
                # If found both segments
                if u_seg is not None and v_seg is not None and u_seg != v_seg:
                    pair = tuple(sorted([u_seg, v_seg]))
                    f_A[pair] += 1
        
        # Compute boundary counts between segments
        for i in range(H):
            for j in range(W):
                if S0[i, j] < 0:
                    continue
                
                current_seg = S0[i, j]
                
                # Check neighbors
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and S0[ni, nj] >= 0:
                        neighbor_seg = S0[ni, nj]
                        if neighbor_seg != current_seg:
                            pair = tuple(sorted([current_seg, neighbor_seg]))
                            boundary_counts[pair] += 1
        
        # Build A relation
        A = set()
        for pair, count in f_A.items():
            boundary_count = boundary_counts.get(pair, 0)
            if boundary_count > 0 and count > self.tau_a * boundary_count:
                A.add(pair)
        
        return A
    
    def _solve_multicut(
        self,
        graph: nx.Graph,
        A: Set[Tuple[int, int]],
        verbose: bool = False
    ) -> Dict[int, int]:
        """
        Solve multi-terminal min-cut problem
        
        Uses iterative min-cut approach as described in Section 3.2
        Returns mapping from original segment to cluster ID
        """
        # Sort A pairs by some heuristic (e.g., number of boundary pixels)
        A_list = list(A)
        
        # Initialize: each segment in its own cluster
        segment_to_cluster = {node: node for node in graph.nodes()}
        
        # Create working graph
        working_graph = graph.copy()
        
        if verbose:
            print(f"  - Solving multicut with {len(A_list)} discontinuity pairs")
        
        # Iteratively apply min-cut for each pair in A
        for idx, (u, v) in enumerate(A_list):
            # Check if u and v are still in the same component
            cluster_u = segment_to_cluster[u]
            cluster_v = segment_to_cluster[v]
            
            if cluster_u == cluster_v:
                # Need to separate them
                # Find all nodes in this cluster
                cluster_nodes = [n for n, c in segment_to_cluster.items() if c == cluster_u]
                
                # Create subgraph
                subgraph = working_graph.subgraph(cluster_nodes).copy()
                
                if not subgraph.has_edge(u, v):
                    continue
                
                # Remove the edge with minimum weight cut
                # Simple approach: just remove the edge between u and v
                if working_graph.has_edge(u, v):
                    working_graph.remove_edge(u, v)
                
                # Update clusters based on connected components
                components = list(nx.connected_components(working_graph))
                
                # Reassign clusters
                for comp_idx, component in enumerate(components):
                    new_cluster = min(component)  # Use minimum node ID as cluster ID
                    for node in component:
                        segment_to_cluster[node] = new_cluster
        
        return segment_to_cluster
    
    def segment(
        self,
        image: np.ndarray,
        S0: np.ndarray,
        discontinuity: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Perform discontinuity-aware segmentation
        
        Args:
            image: Smoothed RGB image (H, W, 3)
            S0: Initial color-based segmentation
            discontinuity: Discontinuity map
            verbose: Print progress
            
        Returns:
            Final segmentation S
        """
        # Build graph
        graph, segment_info = self._build_segment_graph(image, S0)
        
        if verbose:
            print(f"  - Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Compute discontinuity relation
        A = self._compute_discontinuity_relation(S0, discontinuity, segment_info)
        
        if verbose:
            print(f"  - Discontinuity pairs (A): {len(A)}")
        
        # Solve multicut
        segment_to_cluster = self._solve_multicut(graph, A, verbose=verbose)
        
        # Create final segmentation
        S = S0.copy()
        for seg, cluster in segment_to_cluster.items():
            S[S0 == seg] = cluster
        
        # Relabel to consecutive integers
        unique_labels = np.unique(S[S >= 0])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        
        for old_label, new_label in label_map.items():
            S[S0 == old_label] = new_label
        
        return S


def test_segmentation():
    """Test segmentation modules"""
    import matplotlib.pyplot as plt
    
    # Create test image
    img = np.zeros((100, 100, 3))
    img[10:50, 10:50] = [1.0, 0.0, 0.0]  # Red
    img[50:90, 10:50] = [0.0, 1.0, 0.0]  # Green
    img[10:90, 50:90] = [0.0, 0.0, 1.0]  # Blue
    
    # Add some noise
    img += np.random.randn(*img.shape) * 0.05
    img = np.clip(img, 0, 1)
    
    # Color-based segmentation
    segmenter = ColorBasedSegmenter(threshold=15.0)
    labels = segmenter.segment(img)
    
    print(f"Number of segments: {labels.max() + 1}")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(labels, cmap='tab20')
    plt.title('Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Show boundaries
    from skimage import segmentation
    boundaries = segmentation.find_boundaries(labels, mode='inner')
    plt.imshow(img)
    plt.imshow(boundaries, cmap='Reds', alpha=0.5)
    plt.title('Boundaries')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/segmentation_test.png', dpi=150)
    print("Test saved to segmentation_test.png")


if __name__ == '__main__':
    test_segmentation()

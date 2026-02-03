"""
SVG Generation Module
Converts paths and gradient information to SVG format
"""

import numpy as np
from typing import List, Dict, Tuple
from xml.etree import ElementTree as ET


class SVGGenerator:
    """
    Generate SVG from paths and segment information
    """
    
    def __init__(self):
        pass
    
    def generate(
        self,
        image_size: Tuple[int, int],
        paths: List[Dict],
        segments_info: Dict
    ) -> str:
        """
        Generate SVG content
        
        Args:
            image_size: (height, width) of image
            paths: List of path dictionaries
            segments_info: Dictionary with segment fill information
            
        Returns:
            SVG content as string
        """
        height, width = image_size
        
        # Create SVG root
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        
        # Create defs for gradients
        defs = ET.SubElement(svg, 'defs')
        gradient_id_counter = 0
        gradient_ids = {}
        
        # Generate gradient definitions
        for seg_id, info in segments_info.items():
            if info['fill_type'] == 'linear':
                grad_id = f'linear_grad_{gradient_id_counter}'
                gradient_id_counter += 1
                gradient_ids[seg_id] = grad_id
                
                self._create_linear_gradient(defs, grad_id, info)
                
            elif info['fill_type'] == 'radial':
                grad_id = f'radial_grad_{gradient_id_counter}'
                gradient_id_counter += 1
                gradient_ids[seg_id] = grad_id
                
                self._create_radial_gradient(defs, grad_id, info)
        
        # Create paths
        for path in paths:
            seg_id = path['segment_id']
            
            if seg_id not in segments_info:
                continue
            
            info = segments_info[seg_id]
            
            # Create path element
            path_elem = ET.SubElement(svg, 'path')
            
            # Set path data
            path_data = self._points_to_path_data(path['points'], path['closed'])
            path_elem.set('d', path_data)
            
            # Set fill
            if info['fill_type'] == 'constant':
                color = self._rgb_to_hex(info['color'])
                path_elem.set('fill', color)
            elif info['fill_type'] in ['linear', 'radial']:
                grad_id = gradient_ids.get(seg_id)
                if grad_id:
                    path_elem.set('fill', f'url(#{grad_id})')
            
            # No stroke by default
            path_elem.set('stroke', 'none')
        
        # Convert to string
        svg_string = self._prettify_svg(svg)
        
        return svg_string
    
    def _create_linear_gradient(
        self,
        defs: ET.Element,
        grad_id: str,
        info: Dict
    ):
        """Create linear gradient definition"""
        gradient = ET.SubElement(defs, 'linearGradient')
        gradient.set('id', grad_id)
        
        # Set gradient direction
        direction = info['direction']
        
        # Convert direction to x1,y1,x2,y2
        # Direction is a unit vector [dx, dy]
        # We'll set gradient from -direction to +direction
        scale = 100  # Arbitrary scale
        
        x1 = 50 - direction[0] * scale
        y1 = 50 - direction[1] * scale
        x2 = 50 + direction[0] * scale
        y2 = 50 + direction[1] * scale
        
        gradient.set('x1', f'{x1}%')
        gradient.set('y1', f'{y1}%')
        gradient.set('x2', f'{x2}%')
        gradient.set('y2', f'{y2}%')
        gradient.set('gradientUnits', 'userSpaceOnUse')
        
        # Add stops
        stops = info['stops']
        
        if len(stops) == 0:
            return
        
        # Normalize stop positions to [0, 1]
        positions = [s[0] for s in stops]
        min_pos = min(positions)
        max_pos = max(positions)
        
        if max_pos - min_pos > 0:
            norm_positions = [(p - min_pos) / (max_pos - min_pos) for p in positions]
        else:
            norm_positions = [0.5] * len(positions)
        
        for i, (pos, color) in enumerate(zip(norm_positions, [s[1] for s in stops])):
            stop = ET.SubElement(gradient, 'stop')
            stop.set('offset', f'{pos * 100}%')
            stop.set('stop-color', self._rgb_to_hex(color))
    
    def _create_radial_gradient(
        self,
        defs: ET.Element,
        grad_id: str,
        info: Dict
    ):
        """Create radial gradient definition"""
        gradient = ET.SubElement(defs, 'radialGradient')
        gradient.set('id', grad_id)
        
        # Set focal point
        focus = info['focus']
        gradient.set('fx', f'{focus[0]}')
        gradient.set('fy', f'{focus[1]}')
        gradient.set('cx', f'{focus[0]}')
        gradient.set('cy', f'{focus[1]}')
        
        # Set radius (estimate from stops)
        stops = info['stops']
        if len(stops) > 0:
            max_radius = max([s[0] for s in stops])
            gradient.set('r', f'{max_radius}')
        
        gradient.set('gradientUnits', 'userSpaceOnUse')
        
        # Add stops
        if len(stops) == 0:
            return
        
        # Normalize stop positions
        positions = [s[0] for s in stops]
        max_pos = max(positions) if max(positions) > 0 else 1.0
        
        for pos, color in stops:
            stop = ET.SubElement(gradient, 'stop')
            offset = pos / max_pos if max_pos > 0 else 0
            stop.set('offset', f'{offset * 100}%')
            stop.set('stop-color', self._rgb_to_hex(color))
    
    def _points_to_path_data(
        self,
        points: List[List[float]],
        closed: bool = False
    ) -> str:
        """Convert list of points to SVG path data"""
        if len(points) == 0:
            return ""
        
        path_parts = []
        
        # Move to first point
        path_parts.append(f"M {points[0][0]:.2f},{points[0][1]:.2f}")
        
        # Line to subsequent points
        for point in points[1:]:
            path_parts.append(f"L {point[0]:.2f},{point[1]:.2f}")
        
        # Close path
        if closed:
            path_parts.append("Z")
        
        return " ".join(path_parts)
    
    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        """Convert RGB [0,1] to hex color"""
        rgb_int = (np.clip(rgb, 0, 1) * 255).astype(int)
        return f'#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}'
    
    def _prettify_svg(self, elem: ET.Element) -> str:
        """Convert ElementTree to formatted string"""
        # Add XML declaration
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_str += ET.tostring(elem, encoding='unicode')
        
        # Basic formatting (indentation)
        # For proper formatting, could use xml.dom.minidom
        return xml_str


def test_svg_generation():
    """Test SVG generation"""
    # Create test data
    image_size = (100, 100)
    
    paths = [
        {
            'points': [[10, 10], [90, 10], [90, 90], [10, 90]],
            'segment_id': 1,
            'closed': True
        },
        {
            'points': [[30, 30], [70, 30], [70, 70], [30, 70]],
            'segment_id': 2,
            'closed': True
        }
    ]
    
    segments_info = {
        1: {
            'fill_type': 'linear',
            'direction': np.array([1, 0]),
            'stops': [
                (0, np.array([1, 0, 0])),
                (100, np.array([0, 0, 1]))
            ]
        },
        2: {
            'fill_type': 'constant',
            'color': np.array([0, 1, 0])
        }
    }
    
    # Generate SVG
    generator = SVGGenerator()
    svg_content = generator.generate(image_size, paths, segments_info)
    
    # Save
    with open('/home/claude/test_output.svg', 'w') as f:
        f.write(svg_content)
    
    print("Test SVG saved to test_output.svg")
    print("\nSVG Content Preview:")
    print(svg_content[:500] + "...")


if __name__ == '__main__':
    test_svg_generation()

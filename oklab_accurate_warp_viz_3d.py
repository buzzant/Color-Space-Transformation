#!/usr/bin/env python3
"""
Full 3D RGB Cube to OKLAB Warping Visualization
Shows how the complete 3D RGB color cube transforms through each step
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as ps
from matplotlib import cm
import matplotlib.gridspec as gridspec


class AccurateOKLABConverter:
    """Implements the exact OKLAB conversion as specified in the mathematical documentation"""
    
    @staticmethod
    def srgb_to_linear(srgb):
        """Step 1: Gamma decoding (linearization)"""
        srgb = np.asarray(srgb)
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear
    
    @staticmethod
    def linear_to_lms_prime(linear_rgb):
        """Step 2: Linear transformation to cone space (RGB to LMS')"""
        # Matrix from the documentation
        M = np.array([
            [0.41222147, 0.53633255, 0.05144599],
            [0.21190349, 0.68069954, 0.10739698],
            [0.08830246, 0.28171881, 0.63000000]
        ])
        
        # Handle both single colors and arrays
        if linear_rgb.ndim == 1:
            return M @ linear_rgb
        else:
            return np.tensordot(linear_rgb, M.T, axes=(-1, -1))
    
    @staticmethod
    def lms_prime_to_oklab(lms_prime):
        """Step 3: Achieving perceptual uniformity (LMS' to OKLAB)"""
        # Step 3a: Non-linear compression (cube root)
        lms_cubed = np.sign(lms_prime) * np.power(np.abs(lms_prime), 1/3)
        
        # Step 3b: Final linear transformation to opponent axes
        M2 = np.array([
            [0.21045426,  0.79361779, -0.00407204],
            [1.97799849, -2.42859220,  0.45059371],
            [0.02590404,  0.78277177, -0.80867577]
        ])
        
        if lms_cubed.ndim == 1:
            oklab = M2 @ lms_cubed
        else:
            oklab = np.tensordot(lms_cubed, M2.T, axes=(-1, -1))
        
        return oklab
    
    @staticmethod
    def full_conversion(srgb):
        """Complete sRGB to OKLAB conversion"""
        linear = AccurateOKLABConverter.srgb_to_linear(srgb)
        lms_prime = AccurateOKLABConverter.linear_to_lms_prime(linear)
        oklab = AccurateOKLABConverter.lms_prime_to_oklab(lms_prime)
        return oklab


class RGBCubeWarpingVisualizer:
    """Visualizes how the full 3D RGB cube warps through the conversion"""
    
    def __init__(self, resolution=8):
        self.resolution = resolution
        self.converter = AccurateOKLABConverter()
        self.setup_rgb_cube()
        
    def setup_rgb_cube(self):
        """Create the full 3D RGB cube with sampling points"""
        # Create 3D grid in sRGB space
        r = np.linspace(0, 1, self.resolution)
        g = np.linspace(0, 1, self.resolution)
        b = np.linspace(0, 1, self.resolution)
        
        # Create meshgrid
        self.rr, self.gg, self.bb = np.meshgrid(r, g, b, indexing='ij')
        
        # Flatten for easier processing
        self.srgb_points = np.stack([
            self.rr.flatten(),
            self.gg.flatten(),
            self.bb.flatten()
        ], axis=-1)
        
        # Also create edge points for wireframe
        self.create_cube_edges()
        
        # Compute all transformations
        self.compute_all_spaces()
    
    def create_cube_edges(self):
        """Create points along the edges of the RGB cube for wireframe visualization"""
        edge_res = 20  # Points per edge
        edges = []
        
        # 12 edges of a cube
        # Bottom square (z=0)
        edges.append(np.array([[t, 0, 0] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1, t, 0] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1-t, 1, 0] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[0, 1-t, 0] for t in np.linspace(0, 1, edge_res)]))
        
        # Top square (z=1)
        edges.append(np.array([[t, 0, 1] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1, t, 1] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1-t, 1, 1] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[0, 1-t, 1] for t in np.linspace(0, 1, edge_res)]))
        
        # Vertical edges
        edges.append(np.array([[0, 0, t] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1, 0, t] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[1, 1, t] for t in np.linspace(0, 1, edge_res)]))
        edges.append(np.array([[0, 1, t] for t in np.linspace(0, 1, edge_res)]))
        
        self.edge_points_srgb = edges
    
    def compute_all_spaces(self):
        """Compute color values in all intermediate spaces"""
        # Step 1: Linearization
        self.linear_points = self.converter.srgb_to_linear(self.srgb_points)
        
        # Step 2: LMS' cone space
        self.lms_points = self.converter.linear_to_lms_prime(self.linear_points)
        
        # Step 3a: Cube root (perceptual compression)
        self.lms_cubed_points = np.sign(self.lms_points) * np.power(np.abs(self.lms_points), 1/3)
        
        # Step 3b: Final OKLAB
        self.oklab_points = self.converter.lms_prime_to_oklab(self.lms_points)
        
        # Transform edges too
        self.edge_points_linear = []
        self.edge_points_lms = []
        self.edge_points_lms_cubed = []
        self.edge_points_oklab = []
        
        for edge in self.edge_points_srgb:
            linear_edge = self.converter.srgb_to_linear(edge)
            lms_edge = self.converter.linear_to_lms_prime(linear_edge)
            lms_cubed_edge = np.sign(lms_edge) * np.power(np.abs(lms_edge), 1/3)
            oklab_edge = self.converter.lms_prime_to_oklab(lms_edge)
            
            self.edge_points_linear.append(linear_edge)
            self.edge_points_lms.append(lms_edge)
            self.edge_points_lms_cubed.append(lms_cubed_edge)
            self.edge_points_oklab.append(oklab_edge)
    
    def plot_matplotlib_3d(self):
        """Create matplotlib 3D visualization showing all transformation steps"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        positions = [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (2, 3, 4), (2, 3, 5), (2, 3, 6)
        ]
        titles = [
            'Original sRGB Cube',
            'After Gamma Decoding (Linear RGB)',
            'LMS\' Cone Space',
            'After Cube Root Compression',
            'Final OKLAB Space',
            'OKLAB with Color Mapping'
        ]
        
        spaces = [
            self.srgb_points,
            self.linear_points,
            self.lms_points,
            self.lms_cubed_points,
            self.oklab_points,
            self.oklab_points  # Repeated for color version
        ]
        
        edge_spaces = [
            self.edge_points_srgb,
            self.edge_points_linear,
            self.edge_points_lms,
            self.edge_points_lms_cubed,
            self.edge_points_oklab,
            self.edge_points_oklab
        ]
        
        axis_labels = [
            ('R', 'G', 'B'),
            ('Linear R', 'Linear G', 'Linear B'),
            ('L\'', 'M\'', 'S\''),
            ('L\'^(1/3)', 'M\'^(1/3)', 'S\'^(1/3)'),
            ('a*', 'b*', 'L*'),
            ('a*', 'b*', 'L*')
        ]
        
        for idx, (pos, title, points, edges, labels) in enumerate(zip(positions, titles, spaces, edge_spaces, axis_labels)):
            ax = fig.add_subplot(*pos, projection='3d')
            
            # Special handling for OKLAB (different axis ordering)
            if idx >= 4:
                x, y, z = points[:, 1], points[:, 2], points[:, 0]
            else:
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Plot edges (wireframe)
            for edge in edges:
                if idx >= 4:
                    ax.plot(edge[:, 1], edge[:, 2], edge[:, 0], 'k-', alpha=0.3, linewidth=0.5)
                else:
                    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'k-', alpha=0.3, linewidth=0.5)
            
            # Plot points
            if idx == 5:  # Color version
                # Show actual colors
                colors = self.srgb_points
                ax.scatter(x, y, z, c=colors, s=20, alpha=0.7)
            else:
                # Show structure with subtle coloring based on position
                ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.6)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            
            # Adjust viewing angle for better visualization
            ax.view_init(elev=20, azim=45)
            
        plt.suptitle('3D RGB Cube Warping Through sRGB → OKLAB Transformation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_3d(self):
        """Create interactive Plotly visualization"""
        # Create subplots
        fig = ps.make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'sRGB Cube', 'Linear RGB', 'LMS\' Cone Space',
                'After Cube Root', 'OKLAB Space', 'Warping Animation'
            ),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.02,
            vertical_spacing=0.08
        )
        
        # Create color strings for Plotly
        colors_rgb = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) 
                      for r, g, b in self.srgb_points]
        
        # Plot each space
        spaces_data = [
            (self.srgb_points, self.edge_points_srgb, 'sRGB'),
            (self.linear_points, self.edge_points_linear, 'Linear'),
            (self.lms_points, self.edge_points_lms, 'LMS'),
            (self.lms_cubed_points, self.edge_points_lms_cubed, 'Cubed'),
            (self.oklab_points, self.edge_points_oklab, 'OKLAB')
        ]
        
        for idx, (points, edges, name) in enumerate(spaces_data):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # Special handling for OKLAB axes
            if idx == 4:
                x, y, z = points[:, 1], points[:, 2], points[:, 0]
            else:
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors_rgb,
                        opacity=0.6
                    ),
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    text=[f'{name}: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})' 
                          for i in range(len(x))]
                ),
                row=row, col=col
            )
            
            # Add wireframe edges
            for edge in edges:
                if idx == 4:
                    edge_x, edge_y, edge_z = edge[:, 1], edge[:, 2], edge[:, 0]
                else:
                    edge_x, edge_y, edge_z = edge[:, 0], edge[:, 1], edge[:, 2]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='black', width=2),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Add animation in the last subplot
        # This shows selected transformation paths
        sample_indices = np.random.choice(len(self.srgb_points), size=min(100, len(self.srgb_points)), replace=False)
        
        for i in sample_indices:
            # Create path from sRGB to OKLAB
            path_points = np.array([
                self.srgb_points[i],
                self.linear_points[i],
                self.lms_points[i],
                self.oklab_points[i, [1, 2, 0]]  # Reorder for consistency
            ])
            
            fig.add_trace(
                go.Scatter3d(
                    x=path_points[:, 0],
                    y=path_points[:, 1],
                    z=path_points[:, 2],
                    mode='lines+markers',
                    line=dict(
                        color=colors_rgb[i],
                        width=2
                    ),
                    marker=dict(size=4),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D RGB Cube to OKLAB Transformation',
            height=900,
            showlegend=False
        )
        
        # Update scenes for better default view
        for i in range(1, 7):
            fig.update_scenes(
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                row=(i-1)//3 + 1,
                col=(i-1)%3 + 1
            )
        
        fig.show()
    
    def plot_volume_analysis(self):
        """Analyze how the volume and shape of the color space changes"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Calculate convex hull volumes (approximate)
        from scipy.spatial import ConvexHull
        
        spaces = [
            ('sRGB', self.srgb_points),
            ('Linear RGB', self.linear_points),
            ('LMS\'', self.lms_points),
            ('OKLAB', self.oklab_points)
        ]
        
        for idx, (name, points) in enumerate(spaces):
            ax = axes[idx // 2, idx % 2]
            
            # Calculate distances from origin
            distances = np.linalg.norm(points, axis=1)
            
            # Create histogram
            ax.hist(distances, bins=30, alpha=0.7, color=plt.cm.tab10(idx))
            ax.set_title(f'{name} - Distance Distribution from Origin')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Count')
            
            # Add statistics
            ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.3f}')
            ax.axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.3f}')
            ax.legend()
            
            # Calculate and display hull volume if possible
            try:
                hull = ConvexHull(points)
                ax.text(0.95, 0.95, f'Hull Volume: {hull.volume:.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except:
                pass
        
        plt.suptitle('Color Space Volume and Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_corner_analysis(self):
        """Analyze how the 8 corners of the RGB cube transform"""
        # Define the 8 corners of the RGB cube
        corners_srgb = np.array([
            [0, 0, 0],  # Black
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 1, 1],  # White
        ])
        
        corner_names = ['Black', 'Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White']
        
        # Transform corners through all spaces
        corners_linear = self.converter.srgb_to_linear(corners_srgb)
        corners_lms = self.converter.linear_to_lms_prime(corners_linear)
        corners_oklab = self.converter.lms_prime_to_oklab(corners_lms)
        
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        
        # 3D plot showing corner transformations
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot original corners
        ax1.scatter(corners_srgb[:, 0], corners_srgb[:, 1], corners_srgb[:, 2],
                   c=corners_srgb, s=200, edgecolors='black', linewidth=2,
                   alpha=0.9, label='sRGB')
        
        # Add labels
        for i, name in enumerate(corner_names):
            ax1.text(corners_srgb[i, 0], corners_srgb[i, 1], corners_srgb[i, 2],
                    name, fontsize=9)
        
        ax1.set_title('RGB Cube Corners', fontweight='bold')
        ax1.set_xlabel('R')
        ax1.set_ylabel('G')
        ax1.set_zlabel('B')
        
        # Plot OKLAB corners
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(corners_oklab[:, 1], corners_oklab[:, 2], corners_oklab[:, 0],
                   c=corners_srgb, s=200, edgecolors='black', linewidth=2,
                   alpha=0.9, label='OKLAB')
        
        # Add labels
        for i, name in enumerate(corner_names):
            ax2.text(corners_oklab[i, 1], corners_oklab[i, 2], corners_oklab[i, 0],
                    name, fontsize=9)
        
        # Draw connections between corners to show structure
        connections = [
            (0, 1), (0, 2), (0, 4),  # From black
            (7, 6), (7, 5), (7, 3),  # From white
            (1, 3), (1, 5),          # From red
            (2, 4), (2, 6),          # From green
            (4, 5), (3, 6)           # Yellow-Magenta, Blue-Cyan
        ]
        
        for i, j in connections:
            ax2.plot([corners_oklab[i, 1], corners_oklab[j, 1]],
                    [corners_oklab[i, 2], corners_oklab[j, 2]],
                    [corners_oklab[i, 0], corners_oklab[j, 0]],
                    'k-', alpha=0.3, linewidth=0.5)
        
        ax2.set_title('OKLAB Transformed Corners', fontweight='bold')
        ax2.set_xlabel('a*')
        ax2.set_ylabel('b*')
        ax2.set_zlabel('L*')
        
        plt.suptitle('RGB Cube Corner Transformation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    print("="*60)
    print("3D RGB CUBE TO OKLAB WARPING VISUALIZATION")
    print("="*60)
    print("\nThis shows how the complete 3D RGB color cube")
    print("warps through each transformation step to OKLAB.")
    print("\nNote: Using lower resolution for performance.")
    print("Increase resolution parameter for more detail.\n")
    
    # Create visualizer with reasonable resolution
    # (8 means 8x8x8 = 512 points, good balance of detail and performance)
    viz = RGBCubeWarpingVisualizer(resolution=8)
    
    # Generate visualizations
    print("Generating matplotlib 3D visualization...")
    viz.plot_matplotlib_3d()
    
    print("\nAnalyzing RGB cube corner transformations...")
    viz.plot_corner_analysis()
    
    print("\nAnalyzing volume and distribution changes...")
    viz.plot_volume_analysis()
    
    # Interactive 3D (optional - can be slow with many points)
    print("\nGenerating interactive 3D visualization...")
    print("(This will open in your browser - may take a moment)")
    viz.plot_interactive_3d()
    
    print("\n3D Visualization complete!")
    print("\nKey observations:")
    print("- The RGB cube is a perfect cube in sRGB space")
    print("- Gamma decoding slightly deforms the cube (expands dark regions)")
    print("- LMS' transformation rotates and shears the cube")
    print("- Cube root compression further warps the space")
    print("- Final OKLAB space is perceptually uniform but geometrically distorted")
    print("- The 8 corners (primary colors) map to specific positions in OKLAB")


if __name__ == "__main__":
    main()
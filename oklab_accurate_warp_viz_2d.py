#!/usr/bin/env python3
"""
Accurate sRGB to OKLAB Conversion with Step-by-Step Warping Visualization
Shows how space transforms at each stage of the conversion
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as ps
from matplotlib import cm
from matplotlib.patches import Rectangle
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


class WarpingVisualizer:
    """Visualizes the space warping at each step of the conversion"""
    
    def __init__(self, resolution=20):
        self.resolution = resolution
        self.converter = AccurateOKLABConverter()
        self.setup_color_grid()
        
    def setup_color_grid(self):
        """Create a 2D slice through RGB space"""
        # Create grid in sRGB space (fixing blue at 0.5)
        r = np.linspace(0, 1, self.resolution)
        g = np.linspace(0, 1, self.resolution)
        self.rr, self.gg = np.meshgrid(r, g)
        
        # Create RGB array
        self.srgb_grid = np.zeros((self.resolution, self.resolution, 3))
        self.srgb_grid[:, :, 0] = self.rr
        self.srgb_grid[:, :, 1] = self.gg
        self.srgb_grid[:, :, 2] = 0.5  # Fix blue channel
        
        # Compute all intermediate representations
        self.compute_all_spaces()
    
    def compute_all_spaces(self):
        """Compute color values in all intermediate spaces"""
        # Flatten for easier processing
        srgb_flat = self.srgb_grid.reshape(-1, 3)
        
        # Step 1: Linearization
        self.linear_grid = self.converter.srgb_to_linear(srgb_flat)
        self.linear_grid = self.linear_grid.reshape(self.resolution, self.resolution, 3)
        
        # Step 2: LMS' cone space
        self.lms_grid = self.converter.linear_to_lms_prime(self.linear_grid.reshape(-1, 3))
        self.lms_grid = self.lms_grid.reshape(self.resolution, self.resolution, 3)
        
        # Step 3a: Cube root (perceptual compression)
        self.lms_cubed = np.sign(self.lms_grid) * np.power(np.abs(self.lms_grid), 1/3)
        
        # Step 3b: Final OKLAB
        self.oklab_grid = self.converter.lms_prime_to_oklab(self.lms_grid.reshape(-1, 3))
        self.oklab_grid = self.oklab_grid.reshape(self.resolution, self.resolution, 3)
    
    def plot_2d_transformations(self):
        """Create 2D plots showing each transformation step"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Helper function to plot grid lines
        def plot_grid(ax, x_data, y_data, color='black', alpha=0.3):
            # Plot horizontal lines
            for i in range(0, self.resolution, 2):
                ax.plot(x_data[i, :], y_data[i, :], color=color, alpha=alpha, linewidth=0.5)
            # Plot vertical lines
            for j in range(0, self.resolution, 2):
                ax.plot(x_data[:, j], y_data[:, j], color=color, alpha=alpha, linewidth=0.5)
        
        # Row 1: Original sRGB space
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.srgb_grid, origin='lower', extent=[0, 1, 0, 1])
        ax1.set_title('Step 0: sRGB Color Grid', fontweight='bold')
        ax1.set_xlabel('Red')
        ax1.set_ylabel('Green')
        
        ax2 = fig.add_subplot(gs[0, 1])
        plot_grid(ax2, self.rr, self.gg)
        ax2.set_title('sRGB Coordinate Grid')
        ax2.set_xlabel('R')
        ax2.set_ylabel('G')
        ax2.set_aspect('equal')
        
        # Row 2: After linearization
        ax3 = fig.add_subplot(gs[0, 2])
        # Show how gamma decoding stretches dark values
        x = np.linspace(0, 1, 100)
        y_linear = self.converter.srgb_to_linear(x)
        ax3.plot(x, x, 'b--', label='Identity (no change)', alpha=0.5)
        ax3.plot(x, y_linear, 'r-', label='Gamma decoding', linewidth=2)
        ax3.set_title('Step 1: Gamma Decoding Function')
        ax3.set_xlabel('sRGB value')
        ax3.set_ylabel('Linear RGB value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[0, 3])
        plot_grid(ax4, self.linear_grid[:, :, 0], self.linear_grid[:, :, 1])
        ax4.set_title('After Linearization')
        ax4.set_xlabel('Linear R')
        ax4.set_ylabel('Linear G')
        ax4.set_aspect('equal')
        
        # Row 3: LMS' cone space
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(self.lms_grid[:, :, 0].flatten(), 
                   self.lms_grid[:, :, 1].flatten(),
                   c=self.srgb_grid.reshape(-1, 3), s=1)
        ax5.set_title('Step 2: LMS\' Cone Space', fontweight='bold')
        ax5.set_xlabel('L\' (Long wavelength)')
        ax5.set_ylabel('M\' (Medium wavelength)')
        
        ax6 = fig.add_subplot(gs[1, 1])
        plot_grid(ax6, self.lms_grid[:, :, 0], self.lms_grid[:, :, 1])
        ax6.set_title('Grid in LMS\' Space')
        ax6.set_xlabel('L\'')
        ax6.set_ylabel('M\'')
        ax6.set_aspect('equal')
        
        # Row 4: After cube root
        ax7 = fig.add_subplot(gs[1, 2])
        x = np.linspace(0, 1, 100)
        y_cubed = np.power(x, 1/3)
        ax7.plot(x, x, 'b--', label='Identity', alpha=0.5)
        ax7.plot(x, y_cubed, 'g-', label='Cube root', linewidth=2)
        ax7.set_title('Step 3a: Perceptual Compression')
        ax7.set_xlabel('LMS\' value')
        ax7.set_ylabel('Cube root value')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[1, 3])
        plot_grid(ax8, self.lms_cubed[:, :, 0], self.lms_cubed[:, :, 1])
        ax8.set_title('After Cube Root')
        ax8.set_xlabel('L\'^(1/3)')
        ax8.set_ylabel('M\'^(1/3)')
        ax8.set_aspect('equal')
        
        # Row 5: Final OKLAB space
        ax9 = fig.add_subplot(gs[2, 0:2])
        scatter = ax9.scatter(self.oklab_grid[:, :, 1].flatten(), 
                            self.oklab_grid[:, :, 0].flatten(),
                            c=self.srgb_grid.reshape(-1, 3), s=4)
        ax9.set_title('Step 3b: Final OKLAB Space', fontweight='bold', fontsize=14)
        ax9.set_xlabel('a* (Green-Red axis)')
        ax9.set_ylabel('L* (Lightness)')
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(gs[2, 2:4])
        plot_grid(ax10, self.oklab_grid[:, :, 1], self.oklab_grid[:, :, 0], color='red')
        ax10.set_title('Warped Grid in OKLAB Space', fontsize=14)
        ax10.set_xlabel('a* (Green-Red axis)')
        ax10.set_ylabel('L* (Lightness)')
        ax10.set_aspect('equal')
        ax10.grid(True, alpha=0.3)
        
        plt.suptitle('sRGB to OKLAB: Step-by-Step Space Warping', fontsize=16, fontweight='bold')
        plt.show()
    
    def plot_3d_interactive(self):
        """Create interactive 3D visualization using plotly"""
        # Create subplots
        fig = ps.make_subplots(
            rows=2, cols=3,
            subplot_titles=('sRGB Space', 'Linear RGB Space', 'LMS\' Cone Space',
                          'After Cube Root', 'OKLAB Space (Final)', 'Distance Comparison'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Flatten grids for plotting
        colors = self.srgb_grid.reshape(-1, 3)
        colors_rgb = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) 
                      for r, g, b in colors]
        
        # Plot 1: sRGB space
        fig.add_trace(
            go.Scatter3d(
                x=self.srgb_grid[:, :, 0].flatten(),
                y=self.srgb_grid[:, :, 1].flatten(),
                z=self.srgb_grid[:, :, 2].flatten(),
                mode='markers',
                marker=dict(size=3, color=colors_rgb),
                name='sRGB'
            ),
            row=1, col=1
        )
        
        # Plot 2: Linear RGB
        fig.add_trace(
            go.Scatter3d(
                x=self.linear_grid[:, :, 0].flatten(),
                y=self.linear_grid[:, :, 1].flatten(),
                z=self.linear_grid[:, :, 2].flatten(),
                mode='markers',
                marker=dict(size=3, color=colors_rgb),
                name='Linear RGB'
            ),
            row=1, col=2
        )
        
        # Plot 3: LMS'
        fig.add_trace(
            go.Scatter3d(
                x=self.lms_grid[:, :, 0].flatten(),
                y=self.lms_grid[:, :, 1].flatten(),
                z=self.lms_grid[:, :, 2].flatten(),
                mode='markers',
                marker=dict(size=3, color=colors_rgb),
                name='LMS\''
            ),
            row=1, col=3
        )
        
        # Plot 4: After cube root
        fig.add_trace(
            go.Scatter3d(
                x=self.lms_cubed[:, :, 0].flatten(),
                y=self.lms_cubed[:, :, 1].flatten(),
                z=self.lms_cubed[:, :, 2].flatten(),
                mode='markers',
                marker=dict(size=3, color=colors_rgb),
                name='LMS\' Cubed'
            ),
            row=2, col=1
        )
        
        # Plot 5: OKLAB
        fig.add_trace(
            go.Scatter3d(
                x=self.oklab_grid[:, :, 1].flatten(),  # a*
                y=self.oklab_grid[:, :, 2].flatten(),  # b*
                z=self.oklab_grid[:, :, 0].flatten(),  # L*
                mode='markers',
                marker=dict(size=3, color=colors_rgb),
                name='OKLAB'
            ),
            row=2, col=2
        )
        
        # Plot 6: Distance comparison
        # Calculate distances from center point in both spaces
        center_idx = self.resolution // 2
        center_srgb = self.srgb_grid[center_idx, center_idx]
        center_oklab = self.oklab_grid[center_idx, center_idx]
        
        # Calculate distances
        srgb_distances = np.linalg.norm(self.srgb_grid - center_srgb, axis=2)
        oklab_distances = np.linalg.norm(self.oklab_grid - center_oklab, axis=2)
        
        # Normalize for comparison
        srgb_distances_norm = srgb_distances / np.max(srgb_distances)
        oklab_distances_norm = oklab_distances / np.max(oklab_distances)
        
        # Create surface plot showing distance ratio
        distance_ratio = oklab_distances_norm / (srgb_distances_norm + 0.001)
        
        fig.add_trace(
            go.Surface(
                x=self.rr,
                y=self.gg,
                z=distance_ratio,
                colorscale='Viridis',
                name='Distance Warping'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive 3D Visualization of sRGB to OKLAB Transformation',
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='R', row=1, col=1)
        fig.update_yaxes(title_text='G', row=1, col=1)
        
        fig.show()
    
    def plot_distance_preservation(self):
        """Visualize how perceptual distances are preserved in OKLAB"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Select several reference points
        ref_points = [
            (5, 5),    # Dark corner
            (15, 15),  # Mid-range
            (10, 5),   # Different ratio
        ]
        
        for idx, (ref_i, ref_j) in enumerate(ref_points):
            ref_srgb = self.srgb_grid[ref_i, ref_j]
            ref_oklab = self.oklab_grid[ref_i, ref_j]
            
            # Calculate distances in both spaces
            srgb_dist = np.linalg.norm(self.srgb_grid - ref_srgb, axis=2)
            oklab_dist = np.linalg.norm(self.oklab_grid - ref_oklab, axis=2)
            
            # Plot sRGB distances
            im1 = axes[0, idx].imshow(srgb_dist, cmap='viridis', origin='lower')
            axes[0, idx].plot(ref_j, ref_i, 'r*', markersize=10)
            axes[0, idx].set_title(f'sRGB Distance from ({ref_i},{ref_j})')
            plt.colorbar(im1, ax=axes[0, idx])
            
            # Plot OKLAB distances
            im2 = axes[1, idx].imshow(oklab_dist, cmap='plasma', origin='lower')
            axes[1, idx].plot(ref_j, ref_i, 'r*', markersize=10)
            axes[1, idx].set_title(f'OKLAB Distance from ({ref_i},{ref_j})')
            plt.colorbar(im2, ax=axes[1, idx])
        
        plt.suptitle('Distance Preservation: sRGB vs OKLAB', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    print("="*60)
    print("ACCURATE sRGB TO OKLAB WARPING VISUALIZATION")
    print("="*60)
    print("\nThis demonstrates the exact mathematical transformations")
    print("that warp color space to achieve perceptual uniformity.")
    print("\nVisualization modes:")
    print("1. 2D step-by-step transformation plots")
    print("2. Interactive 3D visualization (opens in browser)")
    print("3. Distance preservation comparison")
    
    # Create visualizer
    viz = WarpingVisualizer(resolution=20)
    
    # Show 2D transformations
    print("\nGenerating 2D transformation plots...")
    viz.plot_2d_transformations()
    
    # Show distance preservation
    print("\nGenerating distance preservation comparison...")
    viz.plot_distance_preservation()
    
    # Interactive 3D (optional - comment out if not needed)
    print("\nGenerating interactive 3D visualization...")
    print("(This will open in your default browser)")
    viz.plot_3d_interactive()
    
    print("\nVisualization complete!")
    print("\nKey observations:")
    print("- Gamma decoding expands dark regions (non-linear stretch)")
    print("- Linear RGB to LMS' is a rotation/scaling (linear transform)")
    print("- Cube root compresses bright regions (perceptual uniformity)")
    print("- Final OKLAB space has uniform perceptual distances")


if __name__ == "__main__":
    main()
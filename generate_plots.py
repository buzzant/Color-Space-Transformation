#!/usr/bin/env python3
"""
Generate and save all visualization plots for the README
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

class AccurateOKLABConverter:
    """Implements the exact OKLAB conversion"""
    
    @staticmethod
    def srgb_to_linear(srgb):
        srgb = np.asarray(srgb)
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear
    
    @staticmethod
    def linear_to_lms_prime(linear_rgb):
        M = np.array([
            [0.41222147, 0.53633255, 0.05144599],
            [0.21190349, 0.68069954, 0.10739698],
            [0.08830246, 0.28171881, 0.63000000]
        ])
        if linear_rgb.ndim == 1:
            return M @ linear_rgb
        else:
            return np.tensordot(linear_rgb, M.T, axes=(-1, -1))
    
    @staticmethod
    def lms_prime_to_oklab(lms_prime):
        lms_cubed = np.sign(lms_prime) * np.power(np.abs(lms_prime), 1/3)
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


def generate_3d_cube_transformation():
    """Generate the main 3D RGB cube transformation plot"""
    converter = AccurateOKLABConverter()
    
    # Create RGB cube
    resolution = 8
    r = np.linspace(0, 1, resolution)
    g = np.linspace(0, 1, resolution)
    b = np.linspace(0, 1, resolution)
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    
    srgb_points = np.stack([rr.flatten(), gg.flatten(), bb.flatten()], axis=-1)
    
    # Transform through all stages
    linear_points = converter.srgb_to_linear(srgb_points)
    lms_points = converter.linear_to_lms_prime(linear_points)
    oklab_points = converter.lms_prime_to_oklab(lms_points)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original RGB Cube
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(srgb_points[:, 0], srgb_points[:, 1], srgb_points[:, 2],
               c=srgb_points, s=20, alpha=0.6)
    ax1.set_title('sRGB Cube', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])
    
    # Plot 2: LMS Space
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(lms_points[:, 0], lms_points[:, 1], lms_points[:, 2],
               c=srgb_points, s=20, alpha=0.6)
    ax2.set_title('LMS\' Cone Space', fontsize=14, fontweight='bold')
    ax2.set_xlabel('L\'')
    ax2.set_ylabel('M\'')
    ax2.set_zlabel('S\'')
    
    # Plot 3: OKLAB Space
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(oklab_points[:, 1], oklab_points[:, 2], oklab_points[:, 0],
               c=srgb_points, s=20, alpha=0.6)
    ax3.set_title('OKLAB Space', fontsize=14, fontweight='bold')
    ax3.set_xlabel('a* (Green-Red)')
    ax3.set_ylabel('b* (Yellow-Blue)')
    ax3.set_zlabel('L* (Lightness)')
    
    plt.suptitle('RGB Cube Transformation to OKLAB', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/cube_transformation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/cube_transformation.png")


def generate_step_by_step_2d():
    """Generate 2D step-by-step transformation"""
    converter = AccurateOKLABConverter()
    
    # Create 2D grid
    resolution = 30
    r = np.linspace(0, 1, resolution)
    g = np.linspace(0, 1, resolution)
    rr, gg = np.meshgrid(r, g)
    
    srgb_grid = np.zeros((resolution, resolution, 3))
    srgb_grid[:, :, 0] = rr
    srgb_grid[:, :, 1] = gg
    srgb_grid[:, :, 2] = 0.5  # Fix blue
    
    # Transform
    srgb_flat = srgb_grid.reshape(-1, 3)
    linear_flat = converter.srgb_to_linear(srgb_flat)
    lms_flat = converter.linear_to_lms_prime(linear_flat)
    oklab_flat = converter.lms_prime_to_oklab(lms_flat)
    
    oklab_grid = oklab_flat.reshape(resolution, resolution, 3)
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot grid warping
    ax1 = fig.add_subplot(131)
    ax1.imshow(srgb_grid, origin='lower', extent=[0, 1, 0, 1])
    ax1.set_title('Original RGB Grid', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Red Channel')
    ax1.set_ylabel('Green Channel')
    
    # Plot gamma function
    ax2 = fig.add_subplot(132)
    x = np.linspace(0, 1, 100)
    y_linear = converter.srgb_to_linear(x)
    ax2.plot(x, x, 'b--', label='Identity', alpha=0.5)
    ax2.plot(x, y_linear, 'r-', label='Gamma Decoding', linewidth=2)
    ax2.plot(x, np.power(x, 1/3), 'g-', label='Cube Root', linewidth=2)
    ax2.set_title('Non-linear Transformations', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Input')
    ax2.set_ylabel('Output')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot warped grid
    ax3 = fig.add_subplot(133)
    ax3.scatter(oklab_grid[:, :, 1].flatten(), oklab_grid[:, :, 0].flatten(),
               c=srgb_grid.reshape(-1, 3), s=2)
    ax3.set_title('Warped Grid in OKLAB', fontsize=14, fontweight='bold')
    ax3.set_xlabel('a* (Green-Red)')
    ax3.set_ylabel('L* (Lightness)')
    ax3.set_aspect('equal')
    
    plt.suptitle('2D Visualization of Space Warping', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/2d_warping.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/2d_warping.png")


def generate_corner_analysis():
    """Show how RGB cube corners map to OKLAB"""
    converter = AccurateOKLABConverter()
    
    # Define corners
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
    
    # Transform
    corners_linear = converter.srgb_to_linear(corners_srgb)
    corners_lms = converter.linear_to_lms_prime(corners_linear)
    corners_oklab = converter.lms_prime_to_oklab(corners_lms)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # RGB corners
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(corners_srgb[:, 0], corners_srgb[:, 1], corners_srgb[:, 2],
               c=corners_srgb, s=300, edgecolors='black', linewidth=2)
    for i, name in enumerate(corner_names):
        ax1.text(corners_srgb[i, 0], corners_srgb[i, 1], corners_srgb[i, 2],
                f'  {name}', fontsize=10)
    ax1.set_title('RGB Cube Corners', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    
    # OKLAB corners
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(corners_oklab[:, 1], corners_oklab[:, 2], corners_oklab[:, 0],
               c=corners_srgb, s=300, edgecolors='black', linewidth=2)
    for i, name in enumerate(corner_names):
        ax2.text(corners_oklab[i, 1], corners_oklab[i, 2], corners_oklab[i, 0],
                f'  {name}', fontsize=10)
    ax2.set_title('OKLAB Space Corners', fontsize=14, fontweight='bold')
    ax2.set_xlabel('a*')
    ax2.set_ylabel('b*')
    ax2.set_zlabel('L*')
    
    plt.suptitle('RGB Cube Corners Mapped to OKLAB', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/corner_mapping.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/corner_mapping.png")


def generate_analog_layout_demo():
    """Generate analog layout warping visualization"""
    grid_size = 10
    
    # Create placement grid
    positions = np.zeros((grid_size, grid_size, 2))
    for i in range(grid_size):
        for j in range(grid_size):
            positions[i, j] = [i, j]
    
    # Simulate performance metrics
    def calculate_performance(x, y):
        center = grid_size / 2
        # Thermal hotspot in center
        thermal = np.exp(-((x-center)**2 + (y-center)**2) / 10)
        # Parasitic from edges
        edge_dist = min(x, y, grid_size-x, grid_size-y)
        parasitic = 1 / (edge_dist + 1)
        return thermal + parasitic
    
    # Create warped positions (simplified)
    warped_positions = np.zeros_like(positions)
    for i in range(grid_size):
        for j in range(grid_size):
            perf = calculate_performance(i, j)
            # Warp based on performance
            warped_positions[i, j, 0] = i * (1 + 0.2 * perf)
            warped_positions[i, j, 1] = j * (1 + 0.2 * perf)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original grid
    ax = axes[0]
    for i in range(grid_size):
        ax.plot(positions[i, :, 0], positions[i, :, 1], 'b-', alpha=0.5)
        ax.plot(positions[:, i, 0], positions[:, i, 1], 'b-', alpha=0.5)
    ax.set_title('Original Placement Grid', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Performance heatmap
    ax = axes[1]
    perf_map = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            perf_map[i, j] = calculate_performance(i, j)
    im = ax.imshow(perf_map, cmap='hot', origin='lower')
    ax.set_title('Performance Sensitivity', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im, ax=ax)
    
    # Warped grid
    ax = axes[2]
    for i in range(grid_size):
        ax.plot(warped_positions[i, :, 0], warped_positions[i, :, 1], 'r-', alpha=0.5)
        ax.plot(warped_positions[:, i, 0], warped_positions[:, i, 1], 'r-', alpha=0.5)
    ax.set_title('Performance-Warped Grid', fontsize=14, fontweight='bold')
    ax.set_xlabel('Warped X')
    ax.set_ylabel('Warped Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Analog Layout Space Warping Concept', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/analog_layout_warping.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/analog_layout_warping.png")


# Generate all plots
print("Generating plots for README...")
generate_3d_cube_transformation()
generate_step_by_step_2d()
generate_corner_analysis()
generate_analog_layout_demo()
print("\nAll plots generated successfully!")
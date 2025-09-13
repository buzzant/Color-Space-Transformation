#!/usr/bin/env python3
"""
Analysis of OKLAB space dimensions to understand why it appears squished
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OKLABAnalyzer:
    """Analyze the OKLAB color space to understand its shape"""
    
    @staticmethod
    def srgb_to_linear(srgb):
        """Gamma decoding"""
        srgb = np.asarray(srgb)
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear
    
    @staticmethod
    def linear_to_lms_prime(linear_rgb):
        """RGB to LMS'"""
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
        """LMS' to OKLAB"""
        # Cube root
        lms_cubed = np.sign(lms_prime) * np.power(np.abs(lms_prime), 1/3)
        
        # Final transformation
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
    
    def analyze_extremes(self):
        """Analyze the 8 corners and extremes of the RGB cube in OKLAB"""
        # Define key colors
        colors = {
            'Black': [0, 0, 0],
            'White': [1, 1, 1],
            'Red': [1, 0, 0],
            'Green': [0, 1, 0],
            'Blue': [0, 0, 1],
            'Yellow': [1, 1, 0],
            'Magenta': [1, 0, 1],
            'Cyan': [0, 1, 1],
        }
        
        print("\nRGB Cube Corners in OKLAB Space:")
        print("-" * 60)
        print(f"{'Color':<10} {'L*':<10} {'a*':<10} {'b*':<10}")
        print("-" * 60)
        
        oklab_points = []
        for name, rgb in colors.items():
            linear = self.srgb_to_linear(np.array(rgb))
            lms = self.linear_to_lms_prime(linear)
            oklab = self.lms_prime_to_oklab(lms)
            oklab_points.append(oklab)
            print(f"{name:<10} {oklab[0]:>9.4f} {oklab[1]:>9.4f} {oklab[2]:>9.4f}")
        
        oklab_points = np.array(oklab_points)
        
        # Calculate ranges
        print("\n" + "="*60)
        print("OKLAB Space Ranges:")
        print("-" * 60)
        print(f"L* range: [{oklab_points[:, 0].min():.4f}, {oklab_points[:, 0].max():.4f}] "
              f"(span: {oklab_points[:, 0].max() - oklab_points[:, 0].min():.4f})")
        print(f"a* range: [{oklab_points[:, 1].min():.4f}, {oklab_points[:, 1].max():.4f}] "
              f"(span: {oklab_points[:, 1].max() - oklab_points[:, 1].min():.4f})")
        print(f"b* range: [{oklab_points[:, 2].min():.4f}, {oklab_points[:, 2].max():.4f}] "
              f"(span: {oklab_points[:, 2].max() - oklab_points[:, 2].min():.4f})")
        
        return colors, oklab_points
    
    def sample_full_gamut(self, resolution=10):
        """Sample the full RGB gamut and analyze OKLAB distribution"""
        # Create dense sampling
        r = np.linspace(0, 1, resolution)
        g = np.linspace(0, 1, resolution)
        b = np.linspace(0, 1, resolution)
        
        rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
        rgb_points = np.stack([rr.flatten(), gg.flatten(), bb.flatten()], axis=-1)
        
        # Convert to OKLAB
        linear = self.srgb_to_linear(rgb_points)
        lms = self.linear_to_lms_prime(linear)
        oklab = self.lms_prime_to_oklab(lms)
        
        print("\n" + "="*60)
        print(f"Full Gamut Analysis ({len(oklab)} samples):")
        print("-" * 60)
        print(f"L* range: [{oklab[:, 0].min():.4f}, {oklab[:, 0].max():.4f}] "
              f"(span: {oklab[:, 0].max() - oklab[:, 0].min():.4f})")
        print(f"a* range: [{oklab[:, 1].min():.4f}, {oklab[:, 1].max():.4f}] "
              f"(span: {oklab[:, 1].max() - oklab[:, 1].min():.4f})")
        print(f"b* range: [{oklab[:, 2].min():.4f}, {oklab[:, 2].max():.4f}] "
              f"(span: {oklab[:, 2].max() - oklab[:, 2].min():.4f})")
        
        # Calculate aspect ratios
        L_span = oklab[:, 0].max() - oklab[:, 0].min()
        a_span = oklab[:, 1].max() - oklab[:, 1].min()
        b_span = oklab[:, 2].max() - oklab[:, 2].min()
        
        print("\n" + "="*60)
        print("Aspect Ratios (relative to L* span):")
        print("-" * 60)
        print(f"L* : a* : b* = 1.00 : {a_span/L_span:.2f} : {b_span/L_span:.2f}")
        
        return rgb_points, oklab
    
    def visualize_projections(self):
        """Create projections to understand the 3D shape"""
        # Sample the space
        rgb_points, oklab_points = self.sample_full_gamut(resolution=15)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 3D plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(oklab_points[:, 1], oklab_points[:, 2], oklab_points[:, 0],
                            c=rgb_points, s=2, alpha=0.5)
        ax1.set_xlabel('a* (Green-Red)')
        ax1.set_ylabel('b* (Yellow-Blue)')
        ax1.set_zlabel('L* (Lightness)')
        ax1.set_title('3D OKLAB Space')
        ax1.view_init(elev=20, azim=45)
        
        # Different viewing angles
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(oklab_points[:, 1], oklab_points[:, 2], oklab_points[:, 0],
                   c=rgb_points, s=2, alpha=0.5)
        ax2.set_xlabel('a*')
        ax2.set_ylabel('b*')
        ax2.set_zlabel('L*')
        ax2.set_title('View from Side')
        ax2.view_init(elev=0, azim=0)
        
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.scatter(oklab_points[:, 1], oklab_points[:, 2], oklab_points[:, 0],
                   c=rgb_points, s=2, alpha=0.5)
        ax3.set_xlabel('a*')
        ax3.set_ylabel('b*')
        ax3.set_zlabel('L*')
        ax3.set_title('View from Top')
        ax3.view_init(elev=90, azim=0)
        
        # 2D projections
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(oklab_points[:, 1], oklab_points[:, 0], c=rgb_points, s=1, alpha=0.5)
        ax4.set_xlabel('a* (Green-Red)')
        ax4.set_ylabel('L* (Lightness)')
        ax4.set_title('L* vs a* Projection')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.scatter(oklab_points[:, 2], oklab_points[:, 0], c=rgb_points, s=1, alpha=0.5)
        ax5.set_xlabel('b* (Yellow-Blue)')
        ax5.set_ylabel('L* (Lightness)')
        ax5.set_title('L* vs b* Projection')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.scatter(oklab_points[:, 1], oklab_points[:, 2], c=rgb_points, s=1, alpha=0.5)
        ax6.set_xlabel('a* (Green-Red)')
        ax6.set_ylabel('b* (Yellow-Blue)')
        ax6.set_title('a* vs b* Projection')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('OKLAB Space Shape Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def compare_with_lab(self):
        """Compare OKLAB ranges with typical LAB color space ranges"""
        print("\n" + "="*60)
        print("Comparison with Standard LAB Color Space:")
        print("-" * 60)
        print("Standard CIELAB typical ranges:")
        print("  L*: [0, 100]  (lightness)")
        print("  a*: [-128, 127] (green-red)")
        print("  b*: [-128, 127] (blue-yellow)")
        print("\nOKLAB is designed differently:")
        print("  L*: [0, 1] (normalized lightness)")
        print("  a*: smaller range (perceptually uniform)")
        print("  b*: smaller range (perceptually uniform)")
        print("\nThe 'squished' appearance is CORRECT and intentional!")
        print("OKLAB optimizes for perceptual uniformity, not geometric regularity.")


def main():
    print("="*60)
    print("OKLAB SPACE SHAPE ANALYSIS")
    print("="*60)
    print("\nAnalyzing why OKLAB space appears 'squished'...")
    
    analyzer = OKLABAnalyzer()
    
    # Analyze corner points
    colors, oklab_corners = analyzer.analyze_extremes()
    
    # Sample full gamut
    analyzer.sample_full_gamut(resolution=20)
    
    # Compare with standard LAB
    analyzer.compare_with_lab()
    
    # Visualize projections
    print("\nGenerating visualization of OKLAB shape from multiple angles...")
    analyzer.visualize_projections()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("-" * 60)
    print("YES, the OKLAB space SHOULD appear squished!")
    print("\nReasons:")
    print("1. The L* (lightness) axis has range ~[0, 1]")
    print("2. The a* and b* axes have much smaller ranges (~[-0.4, 0.4])")
    print("3. This is BY DESIGN for perceptual uniformity")
    print("4. The cube root compression contributes but isn't the only factor")
    print("5. The final matrix transformation also affects the shape")
    print("\nThe 'squished' shape ensures that equal distances in OKLAB")
    print("correspond to equal perceptual differences to human vision!")


if __name__ == "__main__":
    main()
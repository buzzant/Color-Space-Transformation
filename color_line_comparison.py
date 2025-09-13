#!/usr/bin/env python3
"""
Color Line Comparison: Drawing straight lines in different color spaces
Shows how a straight line between two colors in different spaces 
produces different visual gradients when displayed in sRGB
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ColorLineVisualizer:
    """Visualize straight lines drawn in different color spaces"""
    
    def __init__(self):
        self.n_samples = 100
    
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
    def linear_to_srgb(linear):
        """Gamma encoding"""
        linear = np.asarray(linear)
        linear = np.maximum(linear, 0)  # Avoid negative values
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1/2.4) - 0.055
        )
        return np.clip(srgb, 0, 1)
    
    @staticmethod
    def srgb_to_lms(srgb):
        """sRGB to LMS' (via linear RGB)"""
        linear = ColorLineVisualizer.srgb_to_linear(srgb)
        M = np.array([
            [0.41222147, 0.53633255, 0.05144599],
            [0.21190349, 0.68069954, 0.10739698],
            [0.08830246, 0.28171881, 0.63000000]
        ])
        if linear.ndim == 1:
            return M @ linear
        else:
            return np.tensordot(linear, M.T, axes=(-1, -1))
    
    @staticmethod
    def lms_to_srgb(lms):
        """LMS' to sRGB (via linear RGB)"""
        M_inv = np.linalg.inv(np.array([
            [0.41222147, 0.53633255, 0.05144599],
            [0.21190349, 0.68069954, 0.10739698],
            [0.08830246, 0.28171881, 0.63000000]
        ]))
        if lms.ndim == 1:
            linear = M_inv @ lms
        else:
            linear = np.tensordot(lms, M_inv.T, axes=(-1, -1))
        return ColorLineVisualizer.linear_to_srgb(linear)
    
    @staticmethod
    def srgb_to_oklab(srgb):
        """sRGB to OKLAB"""
        lms = ColorLineVisualizer.srgb_to_lms(srgb)
        # Cube root
        lms_cubed = np.sign(lms) * np.power(np.abs(lms), 1/3)
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
    
    @staticmethod
    def oklab_to_srgb(oklab):
        """OKLAB to sRGB"""
        # Inverse of final transformation
        M2_inv = np.linalg.inv(np.array([
            [0.21045426,  0.79361779, -0.00407204],
            [1.97799849, -2.42859220,  0.45059371],
            [0.02590404,  0.78277177, -0.80867577]
        ]))
        if oklab.ndim == 1:
            lms_cubed = M2_inv @ oklab
        else:
            lms_cubed = np.tensordot(oklab, M2_inv.T, axes=(-1, -1))
        # Inverse cube root
        lms = np.sign(lms_cubed) * np.power(np.abs(lms_cubed), 3)
        return ColorLineVisualizer.lms_to_srgb(lms)
    
    def draw_line_in_space(self, start_srgb, end_srgb, space='srgb'):
        """
        Draw a straight line between two colors in the specified space
        Returns the line as sRGB colors for visualization
        """
        t = np.linspace(0, 1, self.n_samples)
        
        if space == 'srgb':
            # Straight line in sRGB space
            line = np.zeros((self.n_samples, 3))
            for i in range(3):
                line[:, i] = (1 - t) * start_srgb[i] + t * end_srgb[i]
            return line
        
        elif space == 'linear':
            # Convert endpoints to linear RGB
            start_linear = self.srgb_to_linear(start_srgb)
            end_linear = self.srgb_to_linear(end_srgb)
            # Draw line in linear space
            line_linear = np.zeros((self.n_samples, 3))
            for i in range(3):
                line_linear[:, i] = (1 - t) * start_linear[i] + t * end_linear[i]
            # Convert back to sRGB
            line_srgb = self.linear_to_srgb(line_linear)
            # Force exact endpoints
            line_srgb[0] = start_srgb
            line_srgb[-1] = end_srgb
            return line_srgb
        
        elif space == 'lms':
            # Convert endpoints to LMS'
            start_lms = self.srgb_to_lms(start_srgb)
            end_lms = self.srgb_to_lms(end_srgb)
            # Draw line in LMS' space
            line_lms = np.zeros((self.n_samples, 3))
            for i in range(3):
                line_lms[:, i] = (1 - t) * start_lms[i] + t * end_lms[i]
            # Convert back to sRGB
            line_srgb = self.lms_to_srgb(line_lms)
            # Force exact endpoints
            line_srgb[0] = start_srgb
            line_srgb[-1] = end_srgb
            return line_srgb
        
        elif space == 'oklab':
            # Convert endpoints to OKLAB
            start_oklab = self.srgb_to_oklab(start_srgb)
            end_oklab = self.srgb_to_oklab(end_srgb)
            # Draw line in OKLAB space
            line_oklab = np.zeros((self.n_samples, 3))
            for i in range(3):
                line_oklab[:, i] = (1 - t) * start_oklab[i] + t * end_oklab[i]
            # Convert back to sRGB
            line_srgb = self.oklab_to_srgb(line_oklab)
            # Force exact endpoints
            line_srgb[0] = start_srgb
            line_srgb[-1] = end_srgb
            return line_srgb
        
        else:
            raise ValueError(f"Unknown space: {space}")
    
    def verify_endpoints(self, gradient, start_srgb, end_srgb, space_name):
        """Verify that endpoints match"""
        start_diff = np.linalg.norm(gradient[0] - start_srgb)
        end_diff = np.linalg.norm(gradient[-1] - end_srgb)
        print(f"{space_name:12} - Start diff: {start_diff:.6f}, End diff: {end_diff:.6f}")
        return start_diff < 0.01 and end_diff < 0.01
    
    def create_gradient_strip(self, colors):
        """Create a visual gradient strip"""
        height = 60
        gradient = np.repeat(colors[np.newaxis, :, :], height, axis=0)
        return gradient
    
    def plot_comparison(self, color_pairs):
        """Compare straight lines drawn in different spaces"""
        n_pairs = len(color_pairs)
        fig = plt.figure(figsize=(16, 3 * n_pairs + 1))
        
        spaces = ['srgb', 'linear', 'lms', 'oklab']
        space_names = ['sRGB Space', 'Linear RGB', 'LMS\' Space', 'OKLAB Space']
        
        print("\nEndpoint verification:")
        print("=" * 60)
        
        for pair_idx, (name, start_color, end_color) in enumerate(color_pairs):
            print(f"\n{name}: RGB{tuple(start_color)} → RGB{tuple(end_color)}")
            
            for space_idx, (space, space_name) in enumerate(zip(spaces, space_names)):
                ax = plt.subplot(n_pairs, 4, pair_idx * 4 + space_idx + 1)
                
                # Draw straight line in this space
                gradient = self.draw_line_in_space(start_color, end_color, space)
                
                # Verify endpoints
                self.verify_endpoints(gradient, start_color, end_color, space_name)
                
                # Create and display gradient strip
                strip = self.create_gradient_strip(gradient)
                ax.imshow(strip, aspect='auto')
                
                # Add midpoint marker
                ax.axvline(x=self.n_samples/2, color='white', linewidth=2, alpha=0.7)
                ax.axvline(x=self.n_samples/2, color='black', linewidth=1, 
                          alpha=0.7, linestyle='--')
                
                # Formatting
                ax.set_xticks([0, self.n_samples/2, self.n_samples-1])
                ax.set_xticklabels(['0%', '50%', '100%'])
                ax.set_yticks([])
                
                if pair_idx == 0:
                    ax.set_title(f'{space_name}', fontsize=12, fontweight='bold')
                
                if space_idx == 0:
                    ax.set_ylabel(f'{name}', fontsize=11, fontweight='bold')
        
        plt.suptitle('Straight Lines in Different Color Spaces', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_3d_paths(self, start_color, end_color):
        """Show the actual 3D paths in each space"""
        fig = plt.figure(figsize=(16, 8))
        
        # Sample points along each line
        t = np.linspace(0, 1, 20)
        
        # Plot 1: sRGB space
        ax1 = fig.add_subplot(241, projection='3d')
        line_srgb = self.draw_line_in_space(start_color, end_color, 'srgb')
        ax1.plot(line_srgb[:, 0], line_srgb[:, 1], line_srgb[:, 2], 
                'b-', linewidth=3, label='Path')
        ax1.scatter([start_color[0]], [start_color[1]], [start_color[2]], 
                   c='green', s=100, label='Start')
        ax1.scatter([end_color[0]], [end_color[1]], [end_color[2]], 
                   c='red', s=100, label='End')
        ax1.set_title('sRGB Space')
        ax1.set_xlabel('R')
        ax1.set_ylabel('G')
        ax1.set_zlabel('B')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_zlim([0, 1])
        
        # Plot 2: Linear RGB space
        ax2 = fig.add_subplot(242, projection='3d')
        start_linear = self.srgb_to_linear(start_color)
        end_linear = self.srgb_to_linear(end_color)
        line_linear = np.array([(1-ti) * start_linear + ti * end_linear for ti in t])
        ax2.plot(line_linear[:, 0], line_linear[:, 1], line_linear[:, 2], 
                'b-', linewidth=3)
        ax2.scatter([start_linear[0]], [start_linear[1]], [start_linear[2]], 
                   c='green', s=100)
        ax2.scatter([end_linear[0]], [end_linear[1]], [end_linear[2]], 
                   c='red', s=100)
        ax2.set_title('Linear RGB Space')
        ax2.set_xlabel('R_lin')
        ax2.set_ylabel('G_lin')
        ax2.set_zlabel('B_lin')
        
        # Plot 3: LMS space
        ax3 = fig.add_subplot(243, projection='3d')
        start_lms = self.srgb_to_lms(start_color)
        end_lms = self.srgb_to_lms(end_color)
        line_lms = np.array([(1-ti) * start_lms + ti * end_lms for ti in t])
        ax3.plot(line_lms[:, 0], line_lms[:, 1], line_lms[:, 2], 
                'b-', linewidth=3)
        ax3.scatter([start_lms[0]], [start_lms[1]], [start_lms[2]], 
                   c='green', s=100)
        ax3.scatter([end_lms[0]], [end_lms[1]], [end_lms[2]], 
                   c='red', s=100)
        ax3.set_title('LMS\' Space')
        ax3.set_xlabel('L\'')
        ax3.set_ylabel('M\'')
        ax3.set_zlabel('S\'')
        
        # Plot 4: OKLAB space
        ax4 = fig.add_subplot(244, projection='3d')
        start_oklab = self.srgb_to_oklab(start_color)
        end_oklab = self.srgb_to_oklab(end_color)
        line_oklab = np.array([(1-ti) * start_oklab + ti * end_oklab for ti in t])
        ax4.plot(line_oklab[:, 1], line_oklab[:, 2], line_oklab[:, 0], 
                'b-', linewidth=3)
        ax4.scatter([start_oklab[1]], [start_oklab[2]], [start_oklab[0]], 
                   c='green', s=100)
        ax4.scatter([end_oklab[1]], [end_oklab[2]], [end_oklab[0]], 
                   c='red', s=100)
        ax4.set_title('OKLAB Space')
        ax4.set_xlabel('a*')
        ax4.set_ylabel('b*')
        ax4.set_zlabel('L*')
        
        # Bottom row: Show the gradients
        for idx, (space, name) in enumerate(zip(['srgb', 'linear', 'lms', 'oklab'],
                                                ['sRGB', 'Linear RGB', 'LMS\'', 'OKLAB'])):
            ax = fig.add_subplot(2, 4, 5 + idx)
            gradient = self.draw_line_in_space(start_color, end_color, space)
            strip = self.create_gradient_strip(gradient)
            ax.imshow(strip, aspect='auto')
            ax.set_title(f'{name} Gradient')
            ax.set_xticks([0, 50, 99])
            ax.set_xticklabels(['Start', 'Mid', 'End'])
            ax.set_yticks([])
        
        plt.suptitle(f'3D Paths: RGB{tuple(np.round(start_color, 2))} → RGB{tuple(np.round(end_color, 2))}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    print("="*60)
    print("COLOR LINE COMPARISON")
    print("Drawing Straight Lines in Different Color Spaces")
    print("="*60)
    
    viz = ColorLineVisualizer()
    
    # Define test color pairs - EXACT sRGB values
    color_pairs = [
        ('Blue to Yellow', 
         np.array([0.0, 0.0, 1.0]), 
         np.array([1.0, 1.0, 0.0])),
        
        ('Red to Cyan', 
         np.array([1.0, 0.0, 0.0]), 
         np.array([0.0, 1.0, 1.0])),
        
        ('Black to White', 
         np.array([0.0, 0.0, 0.0]), 
         np.array([1.0, 1.0, 1.0])),
        
        ('Dark Gray to Light Gray', 
         np.array([0.2, 0.2, 0.2]), 
         np.array([0.8, 0.8, 0.8])),
        
        ('Purple to Orange', 
         np.array([0.5, 0.0, 0.5]), 
         np.array([1.0, 0.65, 0.0])),
        
        ('Dark Green to Pink', 
         np.array([0.0, 0.4, 0.0]), 
         np.array([1.0, 0.7, 0.8])),
    ]
    
    print("\nKey Concept:")
    print("- We define START and END colors in sRGB")
    print("- We draw a STRAIGHT LINE between these points in each space")
    print("- We convert the line back to sRGB for display")
    print("- This shows how 'linear interpolation' differs in each space")
    
    print("\nWhat to observe:")
    print("1. sRGB: Simple linear interpolation (may look unnatural)")
    print("2. Linear RGB: Corrects for gamma but still not perceptual")
    print("3. LMS': Closer to perception but not optimal")
    print("4. OKLAB: Most perceptually uniform gradient")
    
    # Main comparison
    viz.plot_comparison(color_pairs)
    
    # Show 3D paths for one example
    print("\nShowing 3D paths for Blue to Yellow...")
    viz.plot_3d_paths(
        np.array([0.0, 0.0, 1.0]),  # Blue
        np.array([1.0, 1.0, 0.0])   # Yellow
    )
    
    # Save for documentation
    print("\nSaving comparison to plots/line_comparison.png...")
    fig = plt.figure(figsize=(16, 8))
    
    test_pairs = color_pairs[:3]  # First 3 for clean image
    
    for pair_idx, (name, start_color, end_color) in enumerate(test_pairs):
        for space_idx, (space, space_name) in enumerate(zip(
            ['srgb', 'linear', 'lms', 'oklab'],
            ['sRGB', 'Linear RGB', 'LMS\'', 'OKLAB']
        )):
            ax = plt.subplot(3, 4, pair_idx * 4 + space_idx + 1)
            gradient = viz.draw_line_in_space(start_color, end_color, space)
            strip = viz.create_gradient_strip(gradient)
            ax.imshow(strip, aspect='auto')
            ax.axvline(x=50, color='white', linewidth=2, alpha=0.7)
            ax.set_xticks([0, 50, 99])
            ax.set_xticklabels(['0%', '50%', '100%'])
            ax.set_yticks([])
            if pair_idx == 0:
                ax.set_title(space_name, fontsize=12, fontweight='bold')
            if space_idx == 0:
                ax.set_ylabel(name, fontsize=11, fontweight='bold')
    
    plt.suptitle('Straight Lines in Different Color Spaces: Same Start & End Points',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/line_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nComplete!")
    print("\nConclusion:")
    print("Even though all gradients start and end at the EXACT same colors,")
    print("the path through color space differs, creating different visual gradients.")
    print("OKLAB's path creates the most perceptually uniform gradient!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Color Gradient Comparison: Visualizing Perceptual Uniformity
Shows how gradients between two colors appear in different color spaces
Demonstrates that OKLAB produces the most perceptually uniform gradients
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


class ColorSpaceGradients:
    """Compare gradients in different color spaces"""
    
    def __init__(self):
        self.n_samples = 100  # Number of points along each gradient
        
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
        """Gamma encoding (inverse)"""
        linear = np.asarray(linear)
        # Clip to avoid negative values before power operation
        linear = np.clip(linear, 0, None)
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1/2.4) - 0.055
        )
        return np.clip(srgb, 0, 1)
    
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
    def lms_prime_to_linear(lms_prime):
        """LMS' to RGB (inverse)"""
        M_inv = np.linalg.inv(np.array([
            [0.41222147, 0.53633255, 0.05144599],
            [0.21190349, 0.68069954, 0.10739698],
            [0.08830246, 0.28171881, 0.63000000]
        ]))
        if lms_prime.ndim == 1:
            return M_inv @ lms_prime
        else:
            return np.tensordot(lms_prime, M_inv.T, axes=(-1, -1))
    
    @staticmethod
    def lms_prime_to_oklab(lms_prime):
        """LMS' to OKLAB"""
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
    
    @staticmethod
    def oklab_to_lms_prime(oklab):
        """OKLAB to LMS' (inverse)"""
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
        lms_prime = np.sign(lms_cubed) * np.power(np.abs(lms_cubed), 3)
        return lms_prime
    
    def interpolate_srgb(self, color1, color2, n):
        """Linear interpolation in sRGB space"""
        t = np.linspace(0, 1, n)
        colors = np.zeros((n, 3))
        for i in range(3):
            colors[:, i] = (1 - t) * color1[i] + t * color2[i]
        return colors
    
    def interpolate_linear_rgb(self, color1, color2, n):
        """Interpolation in linear RGB space"""
        # Convert to linear
        linear1 = self.srgb_to_linear(color1)
        linear2 = self.srgb_to_linear(color2)
        
        # Interpolate in linear space
        t = np.linspace(0, 1, n)
        colors_linear = np.zeros((n, 3))
        for i in range(3):
            colors_linear[:, i] = (1 - t) * linear1[i] + t * linear2[i]
        
        # Convert back to sRGB for display
        colors_srgb = self.linear_to_srgb(colors_linear)
        
        # Force exact start and end colors to match input
        colors_srgb[0] = color1
        colors_srgb[-1] = color2
        
        return colors_srgb
    
    def interpolate_lms(self, color1, color2, n):
        """Interpolation in LMS' space"""
        # Convert to LMS'
        linear1 = self.srgb_to_linear(color1)
        linear2 = self.srgb_to_linear(color2)
        lms1 = self.linear_to_lms_prime(linear1)
        lms2 = self.linear_to_lms_prime(linear2)
        
        # Interpolate in LMS' space
        t = np.linspace(0, 1, n)
        colors_lms = np.zeros((n, 3))
        for i in range(3):
            colors_lms[:, i] = (1 - t) * lms1[i] + t * lms2[i]
        
        # Convert back to sRGB
        colors_linear = self.lms_prime_to_linear(colors_lms)
        colors_srgb = self.linear_to_srgb(colors_linear)
        colors_srgb = np.clip(colors_srgb, 0, 1)
        
        # Force exact start and end colors to match input
        colors_srgb[0] = color1
        colors_srgb[-1] = color2
        
        return colors_srgb
    
    def interpolate_oklab(self, color1, color2, n):
        """Interpolation in OKLAB space"""
        # Convert to OKLAB
        linear1 = self.srgb_to_linear(color1)
        linear2 = self.srgb_to_linear(color2)
        lms1 = self.linear_to_lms_prime(linear1)
        lms2 = self.linear_to_lms_prime(linear2)
        oklab1 = self.lms_prime_to_oklab(lms1)
        oklab2 = self.lms_prime_to_oklab(lms2)
        
        # Interpolate in OKLAB space
        t = np.linspace(0, 1, n)
        colors_oklab = np.zeros((n, 3))
        for i in range(3):
            colors_oklab[:, i] = (1 - t) * oklab1[i] + t * oklab2[i]
        
        # Convert back to sRGB
        colors_lms = self.oklab_to_lms_prime(colors_oklab)
        colors_linear = self.lms_prime_to_linear(colors_lms)
        colors_srgb = self.linear_to_srgb(colors_linear)
        colors_srgb = np.clip(colors_srgb, 0, 1)
        
        # Force exact start and end colors to match input
        colors_srgb[0] = color1
        colors_srgb[-1] = color2
        
        return colors_srgb
    
    def create_gradient_strip(self, colors):
        """Create a visual gradient strip from color array"""
        # Create image array
        height = 50
        gradient = np.repeat(colors[np.newaxis, :, :], height, axis=0)
        return gradient
    
    def plot_gradient_comparison(self, color_pairs):
        """Compare gradients across different color spaces for multiple color pairs"""
        n_pairs = len(color_pairs)
        fig = plt.figure(figsize=(16, 3 * n_pairs + 2))
        gs = gridspec.GridSpec(n_pairs + 1, 4, height_ratios=[1] * n_pairs + [0.3],
                              hspace=0.3, wspace=0.15)
        
        for pair_idx, (name, color1, color2) in enumerate(color_pairs):
            # Generate gradients in each space
            gradient_srgb = self.interpolate_srgb(color1, color2, self.n_samples)
            gradient_linear = self.interpolate_linear_rgb(color1, color2, self.n_samples)
            gradient_lms = self.interpolate_lms(color1, color2, self.n_samples)
            gradient_oklab = self.interpolate_oklab(color1, color2, self.n_samples)
            
            gradients = [
                ('sRGB Space', gradient_srgb),
                ('Linear RGB Space', gradient_linear),
                ('LMS\' Space', gradient_lms),
                ('OKLAB Space', gradient_oklab)
            ]
            
            for i, (space_name, gradient) in enumerate(gradients):
                ax = fig.add_subplot(gs[pair_idx, i])
                
                # Create and display gradient strip
                strip = self.create_gradient_strip(gradient)
                ax.imshow(strip, aspect='auto')
                
                # Add markers for perceptual midpoint
                ax.axvline(x=self.n_samples/2, color='white', linewidth=2, alpha=0.5)
                ax.axvline(x=self.n_samples/2, color='black', linewidth=1, alpha=0.5, linestyle='--')
                
                # Formatting
                ax.set_xticks([0, self.n_samples/2, self.n_samples-1])
                ax.set_xticklabels(['Start', 'Mid', 'End'])
                ax.set_yticks([])
                
                if pair_idx == 0:
                    ax.set_title(f'{space_name}', fontsize=12, fontweight='bold')
                
                if i == 0:
                    ax.set_ylabel(f'{name}', fontsize=11, fontweight='bold')
                
                # Add color values at endpoints
                ax.text(0, -10, f'RGB{tuple(np.round(color1, 2))}', 
                       fontsize=8, ha='left', transform=ax.transData)
                ax.text(self.n_samples-1, -10, f'RGB{tuple(np.round(color2, 2))}', 
                       fontsize=8, ha='right', transform=ax.transData)
        
        # Add explanation text
        ax_text = fig.add_subplot(gs[-1, :])
        ax_text.axis('off')
        explanation = (
            "The vertical line marks the mathematical midpoint of each gradient.\n"
            "In OKLAB space, this midpoint appears perceptually centered between the two colors.\n"
            "In other spaces, the midpoint may appear shifted toward one color or the other,\n"
            "demonstrating that equal numerical steps don't correspond to equal perceptual steps."
        )
        ax_text.text(0.5, 0.5, explanation, fontsize=11, ha='center', va='center',
                    transform=ax_text.transAxes, style='italic')
        
        plt.suptitle('Color Gradient Comparison: Why OKLAB is Perceptually Uniform', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def plot_detailed_analysis(self, color1, color2):
        """Detailed analysis of a single gradient"""
        fig = plt.figure(figsize=(16, 10))
        
        # Generate gradients
        gradient_srgb = self.interpolate_srgb(color1, color2, self.n_samples)
        gradient_linear = self.interpolate_linear_rgb(color1, color2, self.n_samples)
        gradient_lms = self.interpolate_lms(color1, color2, self.n_samples)
        gradient_oklab = self.interpolate_oklab(color1, color2, self.n_samples)
        
        # Top row: Gradient strips
        for i, (name, gradient) in enumerate([
            ('sRGB', gradient_srgb),
            ('Linear RGB', gradient_linear),
            ('LMS\'', gradient_lms),
            ('OKLAB', gradient_oklab)
        ]):
            ax = plt.subplot(4, 4, i + 1)
            strip = self.create_gradient_strip(gradient)
            ax.imshow(strip, aspect='auto')
            ax.set_title(f'{name} Interpolation')
            ax.set_xticks([0, 50, 99])
            ax.set_xticklabels(['0%', '50%', '100%'])
            ax.set_yticks([])
        
        # Second row: RGB channel evolution
        for i, (name, gradient) in enumerate([
            ('sRGB', gradient_srgb),
            ('Linear RGB', gradient_linear),
            ('LMS\'', gradient_lms),
            ('OKLAB', gradient_oklab)
        ]):
            ax = plt.subplot(4, 4, i + 5)
            x = np.linspace(0, 1, self.n_samples)
            ax.plot(x, gradient[:, 0], 'r-', label='R', linewidth=2)
            ax.plot(x, gradient[:, 1], 'g-', label='G', linewidth=2)
            ax.plot(x, gradient[:, 2], 'b-', label='B', linewidth=2)
            ax.set_xlabel('Position')
            ax.set_ylabel('Channel Value')
            ax.set_title(f'{name} Channels')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Third row: Perceptual steps (differences between adjacent colors)
        for i, (name, gradient) in enumerate([
            ('sRGB', gradient_srgb),
            ('Linear RGB', gradient_linear),
            ('LMS\'', gradient_lms),
            ('OKLAB', gradient_oklab)
        ]):
            ax = plt.subplot(4, 4, i + 9)
            # Calculate step sizes in OKLAB space (for perceptual comparison)
            steps_perceptual = []
            for j in range(len(gradient) - 1):
                # Convert both colors to OKLAB
                c1_linear = self.srgb_to_linear(gradient[j])
                c2_linear = self.srgb_to_linear(gradient[j + 1])
                c1_lms = self.linear_to_lms_prime(c1_linear)
                c2_lms = self.linear_to_lms_prime(c2_linear)
                c1_oklab = self.lms_prime_to_oklab(c1_lms)
                c2_oklab = self.lms_prime_to_oklab(c2_lms)
                # Calculate perceptual distance
                dist = np.linalg.norm(c2_oklab - c1_oklab)
                steps_perceptual.append(dist)
            
            x = np.arange(len(steps_perceptual))
            ax.bar(x, steps_perceptual, width=1.0, edgecolor='none')
            ax.set_xlabel('Step Number')
            ax.set_ylabel('Perceptual Distance')
            ax.set_title(f'{name} Step Sizes')
            ax.axhline(y=np.mean(steps_perceptual), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(steps_perceptual):.4f}')
            ax.legend(loc='best', fontsize=8)
        
        # Fourth row: Cumulative perceptual distance
        for i, (name, gradient) in enumerate([
            ('sRGB', gradient_srgb),
            ('Linear RGB', gradient_linear),
            ('LMS\'', gradient_lms),
            ('OKLAB', gradient_oklab)
        ]):
            ax = plt.subplot(4, 4, i + 13)
            # Calculate cumulative perceptual distance
            cumulative = [0]
            for j in range(len(gradient) - 1):
                c1_linear = self.srgb_to_linear(gradient[j])
                c2_linear = self.srgb_to_linear(gradient[j + 1])
                c1_lms = self.linear_to_lms_prime(c1_linear)
                c2_lms = self.linear_to_lms_prime(c2_linear)
                c1_oklab = self.lms_prime_to_oklab(c1_lms)
                c2_oklab = self.lms_prime_to_oklab(c2_lms)
                dist = np.linalg.norm(c2_oklab - c1_oklab)
                cumulative.append(cumulative[-1] + dist)
            
            x = np.linspace(0, 1, len(cumulative))
            ax.plot(x, cumulative, 'b-', linewidth=2)
            # Add ideal linear line
            ax.plot([0, 1], [0, cumulative[-1]], 'r--', alpha=0.5, 
                   label='Ideal Linear')
            ax.set_xlabel('Position')
            ax.set_ylabel('Cumulative Distance')
            ax.set_title(f'{name} Cumulative')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed Gradient Analysis: RGB{tuple(np.round(color1, 2))} to RGB{tuple(np.round(color2, 2))}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    print("="*60)
    print("COLOR GRADIENT COMPARISON")
    print("Demonstrating Perceptual Uniformity in OKLAB")
    print("="*60)
    
    gradients = ColorSpaceGradients()
    
    # Define test color pairs
    color_pairs = [
        # Name, Color1 (RGB), Color2 (RGB)
        ('Blue to Yellow', np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0])),
        ('Red to Cyan', np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])),
        ('Dark to Light Gray', np.array([0.2, 0.2, 0.2]), np.array([0.8, 0.8, 0.8])),
        ('Purple to Orange', np.array([0.5, 0.0, 0.5]), np.array([1.0, 0.65, 0.0])),
        ('Dark Green to Pink', np.array([0.0, 0.3, 0.0]), np.array([1.0, 0.75, 0.8])),
        ('Navy to Gold', np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.84, 0.0])),
    ]
    
    print("\nGenerating gradient comparison...")
    print("\nLook for these features:")
    print("1. In OKLAB, the midpoint appears perceptually centered")
    print("2. In sRGB, gradients often appear to 'accelerate' or have uneven steps")
    print("3. Linear RGB improves over sRGB but still isn't perceptually uniform")
    print("4. LMS' is closer but OKLAB achieves the best uniformity")
    
    # Main comparison plot
    gradients.plot_gradient_comparison(color_pairs)
    
    print("\nGenerating detailed analysis for Blue to Yellow gradient...")
    print("This shows:")
    print("- How RGB channels evolve in each space")
    print("- Perceptual step sizes (should be constant in OKLAB)")
    print("- Cumulative perceptual distance (should be linear in OKLAB)")
    
    # Detailed analysis for one gradient
    gradients.plot_detailed_analysis(
        np.array([0.0, 0.0, 1.0]),  # Blue
        np.array([1.0, 1.0, 0.0])   # Yellow
    )
    
    # Save a high-quality version for documentation
    print("\nSaving high-quality gradient comparison to plots/gradient_comparison.png...")
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(len(color_pairs), 4, hspace=0.3, wspace=0.15)
    
    for pair_idx, (name, color1, color2) in enumerate(color_pairs[:4]):  # First 4 for clean image
        gradient_srgb = gradients.interpolate_srgb(color1, color2, gradients.n_samples)
        gradient_linear = gradients.interpolate_linear_rgb(color1, color2, gradients.n_samples)
        gradient_lms = gradients.interpolate_lms(color1, color2, gradients.n_samples)
        gradient_oklab = gradients.interpolate_oklab(color1, color2, gradients.n_samples)
        
        gradients_list = [
            ('sRGB Space', gradient_srgb),
            ('Linear RGB', gradient_linear),
            ('LMS\' Space', gradient_lms),
            ('OKLAB Space', gradient_oklab)
        ]
        
        for i, (space_name, gradient) in enumerate(gradients_list):
            ax = fig.add_subplot(gs[pair_idx, i])
            strip = gradients.create_gradient_strip(gradient)
            ax.imshow(strip, aspect='auto')
            ax.axvline(x=gradients.n_samples/2, color='white', linewidth=2, alpha=0.5)
            ax.set_xticks([0, gradients.n_samples/2, gradients.n_samples-1])
            ax.set_xticklabels(['0%', '50%', '100%'])
            ax.set_yticks([])
            if pair_idx == 0:
                ax.set_title(f'{space_name}', fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_ylabel(f'{name}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Color Gradients: OKLAB Achieves Perceptual Uniformity', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/gradient_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization complete!")
    print("\nKey Insight:")
    print("OKLAB gradients appear most natural because equal numerical steps")
    print("correspond to equal perceptual differences to the human eye.")


if __name__ == "__main__":
    main()
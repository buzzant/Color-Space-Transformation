# RGB to Oklab Color Space Conversion: A Mathematical Journey

This document outlines the three essential stages required to convert a color from the standard sRGB space to the modern, perceptually uniform Oklab space. Each step involves a specific mathematical transformation that can be understood as a form of "space warping" to align the color data more closely with human vision.

---

## Step 1: Gamma Decoding (Linearization)

### Why is this necessary?
The sRGB values we encounter daily (e.g., in web colors, JPGs) are not linear representations of light intensity. They are **gamma-encoded**, a non-linear compression that efficiently stores color information by dedicating more data bits to darker tones, where the human eye is more sensitive. Before any accurate mathematical operations can be performed, we must reverse this compression to get the "raw," linear light intensity values.

### The Mathematical Operation
This is a non-linear distortion using a piecewise function to decode the gamma. For each sRGB channel value (`C_srgb`) between 0.0 and 1.0, we calculate the corresponding linear value (`C_lin`):

$$
C_{lin} =
\begin{cases}
\frac{C_{srgb}}{12.92} & \text{if } C_{srgb} \le 0.04045 \\
\left( \frac{C_{srgb} + 0.055}{1.055} \right)^{2.4} & \text{if } C_{srgb} > 0.04045
\end{cases}
$$

**Input**: sRGB values `(R_srgb, G_srgb, B_srgb)`
**Output**: Linear RGB values `(R_lin, G_lin, B_lin)`



---

## Step 2: Linear Transformation to Cone Space (RGB to LMS')

### Why is this necessary?
The Linear RGB space is device-centric, describing how much red, green, and blue light a monitor should emit. To move towards a human-centric model, we must translate these values into a space that represents how the three types of cone cells in the human eye (Long, Medium, Short wavelength) respond to light. This transformation is a **change of basis** from the RGB axes to the LMS' axes.

### The Mathematical Operation
This is a pure linear transformation achieved through a 3x3 matrix multiplication. This matrix was empirically derived to model the overlapping spectral responses of the human eye's cones.

$$
\begin{bmatrix} L' \\ M' \\ S' \end{bmatrix} =
\begin{bmatrix}
+0.41222147 & +0.53633255 & +0.05144599 \\
+0.21190349 & +0.68069954 & +0.10739698 \\
+0.08830246 & +0.28171881 & +0.63000000
\end{bmatrix}
\begin{bmatrix} R_{lin} \\ G_{lin} \\ B_{lin} \end{bmatrix}
$$

**Input**: Linear RGB values `(R_lin, G_lin, B_lin)`
**Output**: Cone response values `(L', M', S')`

---

## Step 3: Achieving Perceptual Uniformity (LMS' to Oklab)

This final stage involves two operations to create a space where the geometric distance between two colors directly corresponds to their perceived difference.

### 3a. Non-linear Compression

#### Why is this necessary?
Human perception of stimulus (like brightness) is non-linear, closely following a power law (Stevens' Power Law). Our eyes are more sensitive to changes in dark stimuli than bright stimuli. Applying a cube root function mimics this perceptual response, "stretching" the darker parts of the space and "compressing" the brighter parts to make the distances perceptually uniform.

#### The Mathematical Operation
A cube root is applied to each of the LMS' channels.

$$
\begin{bmatrix} L_{cub} \\ M_{cub} \\ S_{cub} \end{bmatrix} =
\begin{bmatrix} (L')^{1/3} \\ (M')^{1/3} \\ (S')^{1/3} \end{bmatrix}
$$

**Input**: Cone response values `(L', M', S')`
**Output**: Perceptually-scaled cone values `(L_cub, M_cub, S_cub)`

### 3b. Final Linear Transformation (Creating Opponent Axes)

#### Why is this necessary?
After achieving perceptual uniformity, the data is still in a format based on cone stimulation. The human brain, however, interprets color through opponent channels: Light vs. Dark, Red vs. Green, and Yellow vs. Blue. This final matrix multiplication rotates and scales the data into these three intuitive, nearly-orthogonal axes.

#### The Mathematical Operation
A final 3x3 matrix transforms the perceptually-scaled cone data into the final L*, a*, b* coordinates.

$$
\begin{bmatrix} L^* \\ a^* \\ b^* \end{bmatrix} =
\begin{bmatrix}
+0.21045426 & +0.79361779 & -0.00407204 \\
+1.97799849 & -2.42859220 & +0.45059371 \\
+0.02590404 & +0.78277177 & -0.80867577
\end{bmatrix}
\begin{bmatrix} L_{cub} \\ M_{cub} \\ S_{cub} \end{bmatrix}
$$

**Input**: Perceptually-scaled cone values `(L_cub, M_cub, S_cub)`
**Output**: Oklab color values `(L*, a*, b*)`
- **L***: Lightness
- **a***: Red-Green axis
- **b***: Yellow-Blue axis
# Python Phase Retrieval API Documentation

`libspimage` provides a high-level Python interface to configure, execute, and analyze phase retrieval reconstructions. The python backend wraps a high-performance C/CUDA core via SWIG, enabling fast execution (on CPU or GPU) while maintaining an easy-to-use, object-oriented API in Python.

This document describes the python modules, classes, and helper functions associated with phase retrieval.

---

## Table of Contents
1. [Core Modules Overview](#1-core-modules-overview)
2. [Data Layout and Dimensionality Mapping](#2-data-layout-and-dimensionality-mapping)
3. [The `Reconstructor` Class](#3-the-reconstructor-class)
4. [Phase Retrieval Transfer Function (PRTF)](#4-phase-retrieval-transfer-function-prtf)
5. [Patterson Synthesis](#5-patterson-synthesis)
6. [Interactive Python Code Examples](#6-interactive-python-code-examples)

---

## 1. Core Modules Overview

- **[spimage.py](file:///Users/filipe/src/libspimage/src/spimage.py)**: The entrypoint Python module. It exposes both wrapped C functions from the SWIG interface and high-level Python wrappers.
- **[_spimage_reconstructor.py](file:///Users/filipe/src/libspimage/src/_spimage_reconstructor.py)**: Defines the `Reconstructor` class, which serves as the primary object-oriented interface for defining and executing phasing schedules.
- **[_spimage_prtf.py](file:///Users/filipe/src/libspimage/src/_spimage_prtf.py)**: Implements the PRTF calculation, translation matching, and enantiomorph handling.
- **[_spimage_patterson.py](file:///Users/filipe/src/libspimage/src/_spimage_patterson.py)**: Implements the Patterson autocorrelation utility, which is commonly used to generate an initial support constraint.
- **[spimage_pybackend.i](file:///Users/filipe/src/libspimage/src/spimage_pybackend.i)**: The SWIG interface file defining how C structs and arrays map to Python objects.

---

## 2. Data Layout and Dimensionality Mapping

### C-Level Images (`Image` Struct)
Phasing operations are executed in C using the `Image` struct (defined in [image.h](file:///Users/filipe/src/libspimage/include/spimage/image.h)). Key properties include:
- `image`: A complex matrix representing the pixel intensities/amplitudes (`sp_c3matrix`).
- `mask`: An integer matrix representing valid vs. invalid pixels (`sp_i3matrix`).
- `detector`: A pointer to a `Detector` struct containing physical metrics like `wavelength`, `detector_distance`, and `pixel_size`.

### Dimension Mapping to NumPy
Through SWIG typemaps defined in [spimage_pybackend.i](file:///Users/filipe/src/libspimage/src/spimage_pybackend.i), C-level image matrices are converted directly into NumPy arrays:
- **2D Images**: Swapped from C-order `(x, y)` to NumPy `(y, x)` (representing rows and columns respectively, matching standard image plotting conventions like `matplotlib.pyplot.imshow`).
- **3D Images**: Mapped to NumPy `(z, y, x)`.

### FFT Shift Conventions
Phasing algorithm iterations are conducted in **corner-centered (FFT-shifted)** format (where the origin `[0, 0]` is at the corner).
- Input functions (`set_intensities`, `set_mask`) expect unshifted (center-centered) arrays by default and automatically apply `np.fft.fftshift` internally unless called with `shifted=True`.
- Reconstructor output arrays returned to the user are automatically shifted back to center-centered layout.

---

## 3. The `Reconstructor` Class

The `spimage.Reconstructor` class (defined in [_spimage_reconstructor.py](file:///Users/filipe/src/libspimage/src/_spimage_reconstructor.py)) manages a single phasing session.

### Initialization
```python
reconstructor = spimage.Reconstructor(use_gpu=True)
```
- `use_gpu` (bool): Defaults to `True`. If `True`, computations are offloaded to CUDA. If `False`, computation runs on the CPU.

### Data Inputs

#### `.set_intensities(intensities, shifted=False)`
Sets the diffraction pattern intensities to be phased.
- `intensities`: A 2D or 3D numpy array of real-valued intensities. Negative values are clipped to zero, and the amplitude (square root of intensity) is calculated.
- `shifted`: If `False` (default), the intensity pattern is assumed to be centered, and the reconstructor shifts it internally into FFT order. If `True`, the array is assumed to already be in FFT order.

#### `.set_mask(mask, shifted=False)`
Sets the pixel mask of valid/invalid detector pixels.
- `mask`: A 2D or 3D boolean numpy array (same shape as intensities). `True` (or `1`) indicates valid data; `False` (or `0`) indicates masked/invalid data.
- `shifted`: If `False` (default), the mask is shifted internally to match the shifted intensities.

#### `.set_initial_support(**kwargs)`
Defines the starting real-space boundary (support) constraint. Expects exactly **one** of the following keyword arguments:
- `radius`: Radius of a circular support mask (in pixels).
- `area`: Initial support area as a fraction of the total image size (float in `[0, 1]`).
- `support_mask`: Explicit 2D/3D boolean numpy array representing the initial support shape (unshifted).

---

### Algorithm Configuration

#### Phasing Algorithms
Configure which phasing projection algorithm to use, constraints, and iteration count.

- **`.set_phasing_algorithm(type, **kwargs)`**: Clears any existing phasing algorithms and configures a single phasing stage.
- **`.append_phasing_algorithm(type, **kwargs)`**: Appends a phasing stage to the schedule.

**Supported Phasing Types & Arguments:**
- `"er"` (Error Reduction):
  - `number_of_iterations` (int)
- `"hio"` (Hybrid Input-Output) & `"raar"` (Relaxed Averaged Alternating Reflections):
  - `number_of_iterations` (int)
  - `beta_init` (float) & `beta_final` (float): Phasing feedback parameter values at start and end of stage.
- `"diffmap"` (Difference Map):
  - `number_of_iterations` (int)
  - `beta_init` (float) & `beta_final` (float)
  - `gamma1` (float) & `gamma2` (float): Difference map parameters.

**Constraints:**
Phasing algorithms accept a list of strings under the `constraints` keyword parameter to enforce physical features:
- `"enforce_positivity"`: Restricts the real-space object to positive values.
- `"enforce_real"`: Enforces a real-valued object.
- `"enforce_centrosymmetry"`: Enforces centrosymmetry on the object.

*Example:*
```python
reconstructor.append_phasing_algorithm("hio", number_of_iterations=80, beta_init=0.9, beta_final=0.9, constraints=["enforce_positivity", "enforce_real"])
```

#### Support Algorithms
Configure how the real-space support boundary is dynamically updated (e.g. Shrink-Wrap).

- **`.set_support_algorithm(type, number_of_iterations=None, center_image=False, **kwargs)`**: Resets and sets a single support-update strategy.
- **`.append_support_algorithm(type, number_of_iterations, center_image=False, **kwargs)`**: Appends a support-update strategy.

**Supported Support Types & Arguments:**
- `"area"`: Updates support based on a target area fraction.
  - `update_period` (int): Number of iterations between support updates.
  - `blur_init` (float) & `blur_final` (float): Gaussian blur standard deviation (in pixels) used to smooth the reconstruction prior to thresholding.
  - `area_init` (float) & `area_final` (float): Target support area fraction (in `[0, 1]`) at the start and end of this stage.
- `"threshold"`: Updates support based on a relative threshold.
  - `update_period` (int)
  - `blur_init` (float) & `blur_final` (float)
  - `threshold_init` (float) & `threshold_final` (float): Relative intensity threshold value.
- `"static"`: Keeps the support static.
  - `number_of_iterations` (int)

**Image Centering:**
- `center_image` (bool): If `True`, centers the real-space reconstruction (aligning its center of mass to the center of the array) at each support update to prevent drift.

---

### Iteration & Logging Configuration

- **`.set_number_of_iterations(number_of_iterations)`**: Sets the total iteration budget. The sum of iterations in the phasing and support algorithm schedules must match this number.
- **`.set_number_of_outputs_images(n)`**: Sets the number of times image slices (real space, support, Fourier space) are logged and returned.
- **`.set_number_of_outputs_scores(n)`**: Sets the number of times metrics (real/Fourier errors, support size) are logged.
- **`.set_number_of_outputs(n)`**: Helper to set both image and score output counts to `n`.

---

### Execution and Outputs

#### Single Trial: `.reconstruct()`
Runs a single reconstruction starting from a random phase distribution. Returns a dictionary:

| Key | Description | Shape / Type |
|---|---|---|
| `"iteration_index_images"` | Iterations at which image states were captured. | `1D NumPy Array` |
| `"iteration_index_scores"` | Iterations at which metrics were captured. | `1D NumPy Array` |
| `"real_space"` | Complex reconstructed image in real space. | `(num_outputs, Ny, Nx)` complex |
| `"support"` | Boolean mask representing the support. | `(num_outputs, Ny, Nx)` bool |
| `"fourier_space"` | Complex reconstructed model in Fourier space. | `(num_outputs, Ny, Nx)` complex |
| `"mask"` | Boolean array of Fourier masks. | `(num_outputs, Ny, Nx)` bool |
| `"real_error"` | Real-space reconstruction constraint violations. | `(num_outputs,)` float |
| `"fourier_error"` | R-factor between calculated and measured amplitudes. | `(num_outputs,)` float |
| `"support_size"` | Support fraction relative to the total image size. | `(num_outputs,)` float |

#### Multiple Trials: `.reconstruct_loop(Nrepeats, full_output=False)`
Runs the phasing process `Nrepeats` times, each with a different random phase seed. Useful for evaluating reconstruction stability and generating statistics (e.g. PRTF). Returns:

| Key | Description | Shape / Type |
|---|---|---|
| `"real_space_final"` | Final real-space reconstructions for all trials. | `(Nrepeats, Ny, Nx)` complex |
| `"fourier_space_final"` | Final Fourier-space reconstructions. | `(Nrepeats, Ny, Nx)` complex |
| `"support_final"` | Final real-space support masks. | `(Nrepeats, Ny, Nx)` bool |
| `"scores_final"` | Dictionary containing final scores (`"real_error"`, `"fourier_error"`, `"support_size"`) for each repeat. | Dict of `(Nrepeats,)` arrays |
| `"single_outputs"` | A list containing the full history dictionary of each repeat (only returned if `full_output=True`). | `list` of `dict` |

---

## 4. Phase Retrieval Transfer Function (PRTF)

The Phase Retrieval Transfer Function (PRTF) measures the reproducibility of phase retrieval across multiple trials as a function of spatial frequency (resolution). It is defined in [_spimage_prtf.py](file:///Users/filipe/src/libspimage/src/_spimage_prtf.py).

### Usage
```python
output = spimage.prtf(images_rs, supports, translate=True, enantio=True, full_out=False)
```

### Parameters
- `images_rs` (NumPy Array): Shape `(N, Ny, Nx)` of complex real-space reconstructions.
- `supports` (NumPy Array): Shape `(N, Ny, Nx)` of corresponding boolean support masks.
- `translate` (bool): If `True`, translates and aligns the reconstructions to a common spatial reference.
- `enantio` (bool): If `True`, accounts for mirror symmetry ambiguities by testing the enantiomorphic (complex conjugated and mirrored) image.
- `full_out` (bool): If `True`, returns detailed aligned results and radial averages.

### Returns
A dictionary with:
- `"prtf"`: The calculated 2D/3D PRTF array. Values range from `1.0` (perfectly consistent phases) to `0.0` (uncorrelated phases).
- `"super_image"`: The aligned average real-space image.
- If `full_out=True`:
  - `"prtf_r"`: Radial average of the PRTF.
  - `"super_mask"`: Radial average of the support.
  - `"images"`: The aligned real-space images.
  - `"masks"`: The aligned support masks.

---

## 5. Patterson Synthesis

Patterson synthesis calculates the autocorrelation of the electron density (or scattering object) by computing the inverse Fourier transform of the measured diffraction intensities. This is configured via `spimage.patterson` in [_spimage_patterson.py](file:///Users/filipe/src/libspimage/src/_spimage_patterson.py).

### Usage
```python
patterson_map = spimage.patterson(image, mask, floor_cut=None, mask_smooth=1.0, **kwargs)
```
### Parameters
- `image`: 2D NumPy array of diffraction intensities.
- `mask`: 2D NumPy array of mask flags (`1` for valid, `0` for masked).
- `floor_cut`: Minimum intensity cutoff (values below are zeroed).
- `mask_smooth`: Standard deviation of the Gaussian filter applied to smooth mask edges.
- `gauss_damp` / `gauss_damp_sigma`: Applies a Gaussian damping window in Fourier space to suppress high-frequency noise.

---

## 6. Interactive Python Code Examples

### Example 1: Basic Phasing Schedule (HIO followed by ER)
This script demonstrates how to set up intensities and run a single reconstruction using a hybrid phasing schedule.

```python
import numpy as np
import matplotlib.pyplot as plt
import spimage

# 1. Load or generate test data
# For demonstration, we use the internal generator (which uses the classic 'lena' image)
data = spimage.get_test_data()
intensities = data["intensities"]
mask = data["mask"]

# 2. Initialize the Reconstructor
recons = spimage.Reconstructor(use_gpu=True)

# 3. Supply intensities and pixel mask
recons.set_intensities(intensities)
recons.set_mask(mask)

# 4. Define iteration budget and logging frequency
total_iter = 100
recons.set_number_of_iterations(total_iter)
recons.set_number_of_outputs(10)  # Log images and scores 10 times during phasing

# 5. Define an initial support constraint
# We will use a circular support mask that covers 20% of the image area
recons.set_initial_support(area=0.20)

# 6. Configure Phasing Schedule:
# - Stage 1: 80 iterations of HIO with support centering
# - Stage 2: 20 iterations of ER with static support
recons.append_phasing_algorithm("hio", number_of_iterations=80, beta_init=0.9, beta_final=0.9, constraints=["enforce_positivity", "enforce_real"])
recons.append_support_algorithm("area", number_of_iterations=80, update_period=10, blur_init=3.0, blur_final=1.0, area_init=0.20, area_final=0.15, center_image=True)

recons.append_phasing_algorithm("er", number_of_iterations=20, constraints=["enforce_positivity", "enforce_real"])
recons.append_support_algorithm("static", number_of_iterations=20)

# 7. Execute Reconstruction
output = recons.reconstruct()

# 8. Plot the final real-space reconstruction
final_image = np.abs(output["real_space"][-1])
plt.figure(figsize=(6, 5))
plt.imshow(final_image, cmap='gray')
plt.title("Final Reconstructed Object")
plt.colorbar()
plt.show()
```

### Example 2: Phasing Loop and PRTF Resolution Analysis
This script runs multiple independent reconstructions and computes the PRTF to evaluate phase retrieval resolution.

```python
import numpy as np
import matplotlib.pyplot as plt
import spimage

# 1. Load data
data = spimage.get_test_data()
intensities = data["intensities"]
mask = data["mask"]

# 2. Setup reconstructor
recons = spimage.Reconstructor(use_gpu=True)
recons.set_intensities(intensities)
recons.set_mask(mask)
recons.set_number_of_iterations(100)
recons.set_number_of_outputs(5)
recons.set_initial_support(area=0.25)

# Phasing schedule
recons.append_phasing_algorithm("hio", number_of_iterations=80, beta_init=0.9, beta_final=0.9, constraints=["enforce_positivity", "enforce_real"])
recons.append_support_algorithm("area", number_of_iterations=80, update_period=10, blur_init=3.0, blur_final=1.5, area_init=0.25, area_final=0.20, center_image=True)

recons.append_phasing_algorithm("er", number_of_iterations=20, constraints=["enforce_positivity", "enforce_real"])
recons.append_support_algorithm("static", number_of_iterations=20)

# 3. Run multiple trials
num_trials = 5
print(f"Running {num_trials} phase retrieval trials...")
loop_output = recons.reconstruct_loop(Nrepeats=num_trials)

# 4. Compute PRTF
# Note: translate=True and enantio=True align the images/masks to handle drift and mirror ambiguities
prtf_results = spimage.prtf(loop_output["real_space_final"], loop_output["support_final"], translate=True, enantio=True, full_out=True)

# 5. Plot Radial PRTF
radial_prtf = prtf_results["prtf_r"]
plt.figure(figsize=(7, 4))
plt.plot(radial_prtf, marker='o', color='blue')
plt.axhline(y=0.5, color='r', linestyle='--', label="PRTF = 0.5 Threshold")
plt.title("Radial Phase Retrieval Transfer Function (PRTF)")
plt.xlabel("Spatial Frequency (Pixels)")
plt.ylabel("PRTF Value")
plt.legend()
plt.grid(True)
plt.show()
```

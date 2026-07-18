# Raman Alignment Monitor — Phase 1

Standalone live alignment application for the Raman spectrometer. It reuses the existing
`CameraController` but keeps alignment acquisition and analysis separate from `main.py`.

## Configured hardware profile

- Camera: Hamamatsu ORCA-Quest 2 C15550-22UP
- Sensor: 4096 × 2304, 4.6 µm pixel
- Default digital output: 16 bit; 12/8 bit can be selected in the UI
- Cylindrical optics: 300 mm along dispersion, 75 mm perpendicular to dispersion
- Grating: 600 grooves/mm, 500 nm blaze
- Initial reference: Si Stokes peak at 520 cm⁻¹

The Si line is treated as an end-to-end Raman reference. Its measured width includes the intrinsic
Si line shape and therefore is not reported as an absolute instrument line-spread function.

## Run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe alignment_main.py
```

The app starts in simulation mode when the camera or `dcamapi.dll` is unavailable. Install the
Hamamatsu DCAM runtime and driver, connect the camera, and use **Connect ORCA-Quest 2** for hardware
capture.

## Recommended alignment workflow

1. Connect the camera and apply exposure, binning, output bit depth, and vertical ROI.
2. Block the optical input and capture the 20-frame dark average.
3. Unblock the Si signal and start with fixed image levels. Auto levels are for exploration only.
4. Press **Find strongest peak**, then resize the yellow ROI around the Si 520 cm⁻¹ line.
5. Adjust qCMOS, cylindrical optics, and grating while watching:
   - trace tilt and center drift;
   - trace curvature and row FWHM;
   - unrectified projection broadening;
   - peak FWHM, area, and SNR;
   - left/right width, mirror, and second-derivative symmetry;
   - temporal center/FWHM/area stability.
6. Use **Mark A**, **Mark B**, and **Set Best** to retain comparable raw snapshots.
7. Save the session after the final state is held stable.

Do not compare alignment states acquired with different exposure, binning, ROI, readout mode, or
sensor temperature without explicitly accounting for that change. Capturing a new dark is required
after changing the camera ROI or binning.

## Phase 1 metrics

### Detector quality gates

- saturation fraction and detector headroom;
- peak SNR and valid-row ratio;
- vertical clipping margin.

### 2D geometry

- robust trace tilt in pixels per 100 rows;
- active-height center drift;
- quadratic curvature RMS and trace-fit residual;
- median row FWHM and row-width CV;
- row-area CV;
- unrectified projection broadening relative to median row FWHM;
- diagnostic software-rectified projection.

The rectified projection is diagnostic only. Hardware alignment metrics continue to use the
unrectified detector geometry.

### 1D Si peak

- robust asymmetric pseudo-Voigt center, height, area, and FWHM;
- independent left/right HWHM and width-symmetry ratio;
- equal-window area asymmetry and mirrored-profile NRMSE;
- negative second-derivative lobe amplitude, position, and area balance;
- fit NRMSE and Lorentz fraction.

### Stability

- peak-center jitter RMS;
- FWHM and area coefficient of variation over the latest 30 analyzed frames;
- selected live metric trend.

## Session output

Each saved directory can contain:

- `session.json`: hardware profile, analysis configuration, notes, and snapshot summaries;
- `metrics.csv`: time series for all live metrics;
- `snapshots.npz`: raw/corrected A, B, and Best frames, projections, and dark reference.

## Test

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q tests\test_alignment_metrics.py
```

The tests use controlled synthetic frames to verify symmetry, tilt, projection broadening,
curvature, asymmetric width, dark correction, saturation gating, and traceable session export.

## Phase 1 boundaries

- no ML enhancement or deconvolution is used in the metric path;
- no universal overall alignment score is enabled yet;
- no automatic grating/stage motion is performed;
- multi-line field mapping and statistically gated doublet resolution remain Phase 2 work;
- absolute instrument LSF requires a source narrower than the spectrometer response.

import numpy as np
from scipy.constants import c, h, k
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------
# 1. Image & Signal Processing (Denoising & Baseline)
# ---------------------------------------------------------
def denoise_2d(image: np.ndarray, method: str = "median", ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """2D ROI 이미지 노이즈 제거"""
    if image is None or method == "none":
        return image

    img = image.astype(np.float32, copy=False)

    if method == "median":
        ksize = ksize + 1 if ksize % 2 == 0 else ksize  # 홀수 강제 변환
        return median_filter(img, size=(ksize, ksize))
    elif method == "gaussian":
        sigma = max(float(sigma), 0.0)
        return gaussian_filter(img, sigma=(sigma, sigma))

    return image


def apply_gaussian_1d(spectrum: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """1D 스펙트럼 스무딩"""
    if spectrum is None or sigma <= 0:
        return spectrum
    return gaussian_filter1d(spectrum.astype(np.float32, copy=False), sigma=sigma)


def remove_als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """ALS 알고리즘을 이용한 형광 베이스라인 제거 (최적화 완료)"""
    y = np.asarray(y, dtype=np.float64)
    L = y.size
    if L < 3:
        return np.zeros_like(y)

    # D.T @ D 는 반복문 밖에서 한 번만 계산하여 성능 대폭 향상
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc", dtype=np.float64)
    DTD = lam * (D.T @ D)

    w = np.ones(L)
    for _ in range(int(niter)):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + DTD
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    corrected_spectrum = y - z
    corrected_spectrum[corrected_spectrum < 0] = 0  # 음수 방지
    return corrected_spectrum


# ---------------------------------------------------------
# 2. Calibration (Pixel to Raman Shift)
# ---------------------------------------------------------
def calibrate_raman_axis_linear(pixel1, pixel2, raman1, raman2):
    """2점 선형 보정 (Pixel -> Raman Shift)"""
    if pixel2 == pixel1:
        raise ValueError("Pixel positions must be different.")
    slope = (raman2 - raman1) / (pixel2 - pixel1)
    intercept = raman1 - slope * pixel1

    def transform(pixel):
        return slope * pixel + intercept

    return (slope, intercept), transform


def calibrate_raman_axis_quadratic(pixels, wavelengths, laser_wavelength=532.0):
    """3점 2차식 보정 (Pixel -> Wavelength -> Raman Shift)"""
    if len(pixels) != 3 or len(wavelengths) != 3:
        raise ValueError("3개의 pixel과 wavelength 값이 필요합니다.")

    coeffs = np.polyfit(pixels, wavelengths, 2)
    poly = np.poly1d(coeffs)

    def transform(pixel):
        wl_nm = poly(pixel)
        shift = (1e7 / laser_wavelength) - (1e7 / wl_nm)
        return shift

    return coeffs, transform


# ---------------------------------------------------------
# 3. Peak Analysis & Temperature Calculation
# ---------------------------------------------------------
def calculate_integrated_peak(spectrum: np.ndarray, center: int, width: int = 25, x_axis=None) -> float:
    """지정된 중심점 근처의 피크 면적 적분 (로컬 베이스라인 차감 포함)"""
    start = max(center - width, 0)
    end = min(center + width + 1, spectrum.size)
    y_slice = spectrum[start:end]

    if len(y_slice) < 5:
        return 0.0

    left_bg = np.median(y_slice[:5])
    right_bg = np.median(y_slice[-5:])
    baseline = np.linspace(left_bg, right_bg, len(y_slice))

    y_net = y_slice - baseline
    y_net[y_net < 0] = 0

    if x_axis is None:
        return np.sum(y_net)
    else:
        x_slice = x_axis[start:end]
        # numpy 버전 호환성 (np.trapezoid in >=2.0, np.trapz in <2.0)
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz'))
        return abs(trapz_func(y_net, x_slice))


def calculate_snr(spectrum: np.ndarray, center: int, width: int = 60, signal_width: int = 10) -> float:
    """피크 주변 영역을 바탕으로 SNR(신호 대 잡음비) 계산"""
    start = max(center - width, 0)
    end = min(center + width + 1, spectrum.size)
    sig_start = max(center - signal_width, 0)
    sig_end = min(center + signal_width + 1, spectrum.size)

    noise_region = np.concatenate((spectrum[start:sig_start], spectrum[sig_end:end]))

    if len(noise_region) < 2:
        return 0.0

    signal = np.max(spectrum[sig_start:sig_end]) - np.mean(noise_region)
    noise = np.std(noise_region)

    if noise <= 0:
        return float('inf') if signal > 0 else 0.0

    return signal / noise


def calculate_temperature(I_stokes: float, I_antistokes: float, raman_shift_cm_inv: float = 520.0,
                          laser_wavelength_nm: float = 532.0, cal_factor: float = 0.90):
    """Stokes/Anti-Stokes 비율을 이용한 온도 계산"""
    if I_stokes <= 0 or I_antistokes <= 0:
        return None

    ratio = I_stokes / I_antistokes
    laser_shift = 1e7 / laser_wavelength_nm
    cross_section_factor = ((laser_shift - raman_shift_cm_inv) / (laser_shift + raman_shift_cm_inv)) ** 4

    corrected_ratio = ratio / (cross_section_factor * cal_factor)

    if corrected_ratio <= 1:
        return None

    raman_shift_Hz = raman_shift_cm_inv * 100 * c
    T = (h * raman_shift_Hz) / (k * np.log(corrected_ratio))
    return T
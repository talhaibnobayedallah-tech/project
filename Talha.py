import ctypes as ct
import os
import numpy as np


NOISE_GAUSSIAN    = 0
NOISE_SALT_PEPPER = 1
NOISE_UNIFORM     = 2
NOISE_RAYLEIGH    = 3
NOISE_ERLANG      = 4
NOISE_EXPONENTIAL = 5
NOISE_PERIODIC    = 6

_NOISE_NAMES = {
    NOISE_GAUSSIAN:    "gaussian",
    NOISE_SALT_PEPPER: "salt_pepper",
    NOISE_UNIFORM:     "uniform",
    NOISE_RAYLEIGH:    "rayleigh",
    NOISE_ERLANG:      "erlang",
    NOISE_EXPONENTIAL: "exponential",
    NOISE_PERIODIC:    "periodic",
}



class _CImage(ct.Structure):
    _fields_ = [
        ("im_prt",   ct.POINTER(ct.c_ubyte)),
        ("width",    ct.c_int),
        ("height",   ct.c_int),
        ("channels", ct.c_int),
    ]

_CImagePtr = ct.POINTER(_CImage)



class TalhaLib:
    def __init__(self, so_path: str = "./talha.so"):
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library not found: {so_path}")
        self._lib = ct.CDLL(os.path.abspath(so_path))
        self._bind()


    def _bind(self):
        L = self._lib

        def fn(name, restype, *argtypes):
            f = getattr(L, name)
            f.restype  = restype
            f.argtypes = list(argtypes)

        P  = _CImagePtr
        ub = ct.c_ubyte
        i  = ct.c_int
        f  = ct.c_float
        s  = ct.c_char_p
        pf = ct.POINTER(ct.c_float)
        pi = ct.POINTER(ct.c_int)

        # I/O
        fn("readImage",  P, s, i)
        fn("writeImage", None, s, P, i)
        fn("freeImage",  None, P)

        # Math ops
        fn("add_images", P, P, P)
        fn("sub_images", P, P, P)
        fn("mul_images", P, P, P)
        fn("div_images", P, P, P)

        # Generic convolution & histogram
        fn("convolve",   P, P, pf, i)
        fn("histogram",  pi, P)
        fn("freeHist",   None, pi)

        # DCT (matrix method)
        fn("apply_dct",  P, P)

        # ---- Spatial filters ----
        fn("arithmetic_mean_filter",    P, P, i)
        fn("geometric_mean_filter",     P, P, i)
        fn("harmonic_mean_filter",      P, P, i)
        fn("contraharmonic_mean_filter", P, P, i, f)

        fn("median_filter",         P, P, i)
        fn("max_filter",            P, P, i)
        fn("min_filter",            P, P, i)
        fn("midpoint_filter",       P, P, i)
        fn("alpha_trimmed_filter",  P, P, i, i)
        fn("adaptive_median_filter", P, P, i, i)

        fn("laplacian_sharpen_filter",  P, P)
        fn("sobel_filter",              P, P)
        fn("prewitt_filter",            P, P)
        fn("unsharp_masking_filter",    P, P, i, f)

        # ---- Frequency filters ----
        fn("ideal_low_pass_filter",         P, P, f)
        fn("butterworth_low_pass_filter",   P, P, f, i)
        fn("gaussian_low_pass_filter",      P, P, f)

        fn("ideal_high_pass_filter",        P, P, f)
        fn("butterworth_high_pass_filter",  P, P, f, i)
        fn("gaussian_high_pass_filter",     P, P, f)

        fn("gaussian_band_pass_filter",     P, P, f, f)
        fn("gaussian_band_reject_filter",   P, P, f, f)

        fn("gaussian_notch_reject_filter",  P, P, pf, i, f)
        fn("gaussian_notch_pass_filter",    P, P, pf, i, f)

        # Auto dispatch
        fn("auto_filter", P, P, i)

    # ------------------------------------------------------------
    # Pythonic wrappers
    # ------------------------------------------------------------

    def read(self, path: str, channels: int = 1) -> _CImagePtr:
        ptr = self._lib.readImage(path.encode(), channels)
        if not ptr:
            raise ValueError(f"Cannot load image: {path}")
        return ptr

    def write(self, img: _CImagePtr, path: str, quality: int = 90):
        self._lib.writeImage(path.encode(), img, quality)

    def free(self, img: _CImagePtr):
        self._lib.freeImage(img)

    # Pointer → numpy array (view, no copy)
    def to_numpy(self, img: _CImagePtr) -> np.ndarray:
        p   = img.contents
        sz  = p.width * p.height * p.channels
        buf = (ct.c_ubyte * sz).from_address(ct.addressof(p.im_prt.contents))
        arr = np.frombuffer(buf, dtype=np.uint8)
        return arr.reshape((p.height, p.width, p.channels)).squeeze()

    # ---- Math ops ----
    def add(self, a, b):   return self._lib.add_images(a, b)
    def sub(self, a, b):   return self._lib.sub_images(a, b)
    def mul(self, a, b):   return self._lib.mul_images(a, b)
    def div(self, a, b):   return self._lib.div_images(a, b)

    def histogram(self, img) -> list:
        ptr = self._lib.histogram(img)
        h   = [ptr[i] for i in range(256)]
        self._lib.freeHist(ptr)
        return h

    def dct(self, img): return self._lib.apply_dct(img)

    # ---- Spatial ----
    def arithmetic_mean(self, img, fs=5):
        return self._lib.arithmetic_mean_filter(img, fs)

    def geometric_mean(self, img, fs=5):
        return self._lib.geometric_mean_filter(img, fs)

    def harmonic_mean(self, img, fs=5):
        return self._lib.harmonic_mean_filter(img, fs)

    def contraharmonic_mean(self, img, fs=5, Q=1.0):
        return self._lib.contraharmonic_mean_filter(img, fs, ct.c_float(Q))

    def median(self, img, fs=5):
        return self._lib.median_filter(img, fs)

    def max_filt(self, img, fs=5):
        return self._lib.max_filter(img, fs)

    def min_filt(self, img, fs=5):
        return self._lib.min_filter(img, fs)

    def midpoint(self, img, fs=5):
        return self._lib.midpoint_filter(img, fs)

    def alpha_trimmed(self, img, fs=5, d=2):
        return self._lib.alpha_trimmed_filter(img, fs, d)

    def adaptive_median(self, img, fs=3, max_fs=11):
        return self._lib.adaptive_median_filter(img, fs, max_fs)

    def laplacian(self, img):
        return self._lib.laplacian_sharpen_filter(img)

    def sobel(self, img):
        return self._lib.sobel_filter(img)

    def prewitt(self, img):
        return self._lib.prewitt_filter(img)

    def unsharp(self, img, ks=5, alpha=1.5):
        return self._lib.unsharp_masking_filter(img, ks, ct.c_float(alpha))

    # ---- Frequency ----
    def ideal_lpf(self, img, cutoff):
        return self._lib.ideal_low_pass_filter(img, ct.c_float(cutoff))

    def butterworth_lpf(self, img, cutoff, order=2):
        return self._lib.butterworth_low_pass_filter(img, ct.c_float(cutoff), order)

    def gaussian_lpf(self, img, cutoff):
        return self._lib.gaussian_low_pass_filter(img, ct.c_float(cutoff))

    def ideal_hpf(self, img, cutoff):
        return self._lib.ideal_high_pass_filter(img, ct.c_float(cutoff))

    def butterworth_hpf(self, img, cutoff, order=2):
        return self._lib.butterworth_high_pass_filter(img, ct.c_float(cutoff), order)

    def gaussian_hpf(self, img, cutoff):
        return self._lib.gaussian_high_pass_filter(img, ct.c_float(cutoff))

    def gaussian_bpf(self, img, low_cut, high_cut):
        return self._lib.gaussian_band_pass_filter(
            img, ct.c_float(low_cut), ct.c_float(high_cut))

    def gaussian_brf(self, img, low_cut, high_cut):
        return self._lib.gaussian_band_reject_filter(
            img, ct.c_float(low_cut), ct.c_float(high_cut))

    def notch_reject(self, img, centers, radius):
        """centers: list of (u,v) tuples in shifted freq coords."""
        flat = [x for uv in centers for x in uv]
        arr  = (ct.c_float * len(flat))(*flat)
        return self._lib.gaussian_notch_reject_filter(
            img, arr, len(centers), ct.c_float(radius))

    def notch_pass(self, img, centers, radius):
        flat = [x for uv in centers for x in uv]
        arr  = (ct.c_float * len(flat))(*flat)
        return self._lib.gaussian_notch_pass_filter(
            img, arr, len(centers), ct.c_float(radius))

    def auto_filter(self, img, noise_type: int):
        return self._lib.auto_filter(img, noise_type)


# ================================================================
# NOISE DETECTION
# Pure Python / NumPy — no C calls, no scipy required.
# ================================================================

def _mean_blur(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple box-blur using numpy sliding windows (no scipy)."""
    pad = size // 2
    padded = np.pad(arr.astype(float), pad, mode="edge")
    out = np.zeros_like(arr, dtype=float)
    for dr in range(size):
        for dc in range(size):
            out += padded[dr: dr + arr.shape[0], dc: dc + arr.shape[1]]
    return out / (size * size)


def _skewness(x: np.ndarray) -> float:
    """Fisher skewness of a 1-D array."""
    mu  = x.mean()
    std = x.std()
    if std < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def _excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (normal = 0)."""
    mu  = x.mean()
    std = x.std()
    if std < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4)) - 3.0


def detect_noise(image: np.ndarray) -> tuple:
    """
    Estimate the dominant noise type in a grayscale uint8 image.

    Returns
    -------
    (noise_name: str, noise_code: int)

    Algorithm
    ---------
    1. Salt & Pepper — check fraction of extreme (0,255) pixels.
    2. Periodic      — FFT magnitude: presence of strong off-DC spikes.
    3. Statistical   — analyse residual (img − mean_blur) distribution:
         skewness and excess kurtosis classify the remaining types.

    Noise model → distribution → expected stats (approx):
      Gaussian    symmetric,  skew≈0,  excess_kurt≈0
      Uniform     symmetric,  skew≈0,  excess_kurt≈-1.2
      Rayleigh    right-tail, skew≈0.63
      Erlang      right-tail, skew in (0.63, 2)
      Exponential right-tail, skew≈2,  excess_kurt≈6
    """
    img = image.astype(float)

    # ----------------------------------------------------------
    # 1. Salt & Pepper: excessive pixels clamped at 0 or 255
    # ----------------------------------------------------------
    total      = image.size
    n_zeros    = int(np.sum(image == 0))
    n_max      = int(np.sum(image == 255))
    sp_ratio   = (n_zeros + n_max) / total

    if sp_ratio > 0.02:          # >2 % of pixels at extremes
        return "salt_pepper", NOISE_SALT_PEPPER

    # ----------------------------------------------------------
    # 2. Periodic: look for sharp non-DC spikes in the spectrum
    #    We use a log-magnitude map; periodic noise creates
    #    peaks that are >> the spectral mean.
    # ----------------------------------------------------------
    F         = np.fft.fftshift(np.fft.fft2(img))
    mag       = np.abs(F)

    # Zero out DC neighbourhood (7×7 around centre)
    h, w      = mag.shape
    ch, cw    = h // 2, w // 2
    mag[ch-3:ch+4, cw-3:cw+4] = 0.0

    mean_mag  = mag.mean()
    max_mag   = mag.max()

    # A strong isolated spike relative to the spectral floor
    if mean_mag > 0 and max_mag > 12.0 * mean_mag:
        return "periodic", NOISE_PERIODIC

    # ----------------------------------------------------------
    # 3. Statistical analysis of the noise residual
    # ----------------------------------------------------------
    smoothed  = _mean_blur(img, size=3)
    residual  = (img - smoothed).ravel()

    skew      = _skewness(residual)
    ex_kurt   = _excess_kurtosis(residual)

    # Fraction of residual values that are positive
    pos_frac  = float(np.mean(residual > 0))

    # -- Positive-only (right-skewed) noise families --
    # These arise from multiplicative / one-sided noise models.
    if pos_frac > 0.75 or skew > 0.4:
        if skew > 1.5:                        # Exponential: skew≈2
            return "exponential", NOISE_EXPONENTIAL
        elif skew > 0.8:                      # Erlang/Gamma: skew~1
            return "erlang", NOISE_ERLANG
        else:                                 # Rayleigh: skew≈0.63
            return "rayleigh", NOISE_RAYLEIGH

    # -- Symmetric noise families --
    if ex_kurt < -0.5:                        # Uniform: ex_kurt≈-1.2
        return "uniform", NOISE_UNIFORM

    return "gaussian", NOISE_GAUSSIAN         # default / Gaussian


# ================================================================
# auto_denoise — full pipeline: detect → filter → return result
# ================================================================

def auto_denoise(lib: TalhaLib, img_ptr) -> tuple:
    """
    Detect noise in `img_ptr`, call the appropriate C filter, return
    (filtered_ptr, noise_name, noise_code).

    The caller owns the returned pointer and must call lib.free() on it.
    """
    arr = lib.to_numpy(img_ptr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]          # use first channel for detection

    noise_name, noise_code = detect_noise(arr)
    print(f"[noise_pipeline] Detected noise: {noise_name} (code={noise_code})")

    filtered = lib.auto_filter(img_ptr, noise_code)
    return filtered, noise_name, noise_code


# ================================================================
# Quick self-test (run: python noise_pipeline.py image.jpg out.jpg)
# ================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("python3 Talha.py input.jpg output.jpg [so_path] [noise]")
        sys.exit(1)

    so = sys.argv[3] if len(sys.argv) > 3 else "./talha.so"

    noise_map = {
        "g":  NOISE_GAUSSIAN,
        "sp": NOISE_SALT_PEPPER,
        "u":  NOISE_UNIFORM,
        "r":  NOISE_RAYLEIGH,
        "e":  NOISE_ERLANG,
        "ex": NOISE_EXPONENTIAL,
        "p":  NOISE_PERIODIC,
    }

    lib = TalhaLib(so)
    img = lib.read(sys.argv[1], channels=1)

    # manual noise type
    if len(sys.argv) > 4:
        key = sys.argv[4].lower()

        if key not in noise_map:
            print(f"Unknown noise type: {key}")
            sys.exit(1)

        code = noise_map[key]
        filtered = lib.auto_filter(img, code)

        print(f"[manual] Using noise type: {key}")

    else:
        filtered, name, code = auto_denoise(lib, img)
        print(f"[auto] Detected noise: {name}")

    lib.write(filtered, sys.argv[2])
    print(f"[done] Saved to {sys.argv[2]}")

    lib.free(img)
    lib.free(filtered)

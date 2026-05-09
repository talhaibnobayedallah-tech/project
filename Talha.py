import ctypes as c
import os

# --- Load the Shared Library ---
lib_path = os.path.abspath("./talha.so")
lib = c.CDLL(lib_path)

# --- Define C-Struct ---
class CImage(c.Structure):
    _fields_ = [
        ("im_prt", c.POINTER(c.c_ubyte)),
        ("width", c.c_int),
        ("height", c.c_int),
        ("channels", c.c_int)
    ]

# --- Setup C-Function Signatures ---
lib.readImage.restype = c.POINTER(CImage)
lib.readImage.argtypes = [c.c_char_p, c.c_int]

lib.writeImage.restype = None
lib.writeImage.argtypes = [c.c_char_p, c.POINTER(CImage), c.c_int]

lib.freeImage.restype = None
lib.freeImage.argtypes = [c.POINTER(CImage)]

lib.add_images.restype = c.POINTER(CImage)
lib.add_images.argtypes = [c.POINTER(CImage), c.POINTER(CImage)]

lib.sub_images.restype = c.POINTER(CImage)
lib.sub_images.argtypes = [c.POINTER(CImage), c.POINTER(CImage)]

lib.mul_images.restype = c.POINTER(CImage)
lib.mul_images.argtypes = [c.POINTER(CImage), c.POINTER(CImage)]

lib.div_images.restype = c.POINTER(CImage)
lib.div_images.argtypes = [c.POINTER(CImage), c.POINTER(CImage)]

lib.convolve.restype = c.POINTER(CImage)
lib.convolve.argtypes = [c.POINTER(CImage), c.POINTER(c.c_float), c.c_int]

lib.histogram.restype = c.POINTER(c.c_int)
lib.histogram.argtypes = [c.POINTER(CImage)]

lib.freeHist.restype = None
lib.freeHist.argtypes = [c.POINTER(c.c_int)]

lib.apply_dct.restype = c.POINTER(CImage)
lib.apply_dct.argtypes = [c.POINTER(CImage)]


# --- Python Interface Class ---
class TalhaImage:
    def __init__(self, path_or_ptr, channels=1):
        if isinstance(path_or_ptr, str):
            # Read from file
            self.ptr = lib.readImage(path_or_ptr.encode('utf-8'), channels)
            if not self.ptr:
                raise ValueError(f"Failed to load image: {path_or_ptr}")
        else:
            # Init from existing pointer (used internally)
            self.ptr = path_or_ptr

    @property
    def width(self):
        return self.ptr.contents.width

    @property
    def height(self):
        return self.ptr.contents.height

    @property
    def channels(self):
        return self.ptr.contents.channels

    def save(self, path: str, quality: int = 90):
        lib.writeImage(path.encode('utf-8'), self.ptr, quality)

    # --- Operator Overloads ---
    def __add__(self, other):
        return TalhaImage(lib.add_images(self.ptr, other.ptr))

    def __sub__(self, other):
        return TalhaImage(lib.sub_images(self.ptr, other.ptr))

    def __mul__(self, other):
        return TalhaImage(lib.mul_images(self.ptr, other.ptr))

    def __truediv__(self, other):
        return TalhaImage(lib.div_images(self.ptr, other.ptr))

    # --- Filters and Transforms ---
    def convolve(self, kernel_list, kernel_size: int):
        # Convert Python list to C-compatible float array
        FloatArray = c.c_float * len(kernel_list)
        c_kernel = FloatArray(*kernel_list)
        
        res_ptr = lib.convolve(self.ptr, c_kernel, kernel_size)
        return TalhaImage(res_ptr)

    def histogram(self):
        # Returns a standard python list of 256 integers
        hist_ptr = lib.histogram(self.ptr)
        hist_list = [hist_ptr[i] for i in range(256)]
        lib.freeHist(hist_ptr)
        return hist_list

    def dct(self):
        # Warning: For small images only, purely forward logic O(N^4)
        print("Calculating DCT... (This might take a while for large images)")
        res_ptr = lib.apply_dct(self.ptr)
        return TalhaImage(res_ptr)

    # --- Memory Management ---
    def __del__(self):
        # Automatically frees memory when Python garbage collects the object
        if self.ptr:
            lib.freeImage(self.ptr)



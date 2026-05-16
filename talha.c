/*
 * talha.c — Complete Image Processing Library
 *
 * Includes:
 *   - Image I/O and basic math ops (original)
 *   - DCT via matrix method  (single flat loop for matrix generation)
 *   - All spatial filters    (mean family, order-stat family, sharpening)
 *   - Cooley-Tukey FFT       (radix-2, iterative)
 *   - All frequency filters  (LPF, HPF, BPF, BRF, Notch — Ideal/BW/Gaussian)
 *   - auto_filter()          (dispatch by noise-type code from Python)
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o talha.so talha.c -lm
 *
 * All filters expect a SINGLE-CHANNEL (grayscale) Image*.
 * Outputs are always single-channel, normalised to [0,255].
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
   NOISE-TYPE CODES  (must match noise_pipeline.py)
   ================================================================ */
#define NOISE_GAUSSIAN    0
#define NOISE_SALT_PEPPER 1
#define NOISE_UNIFORM     2
#define NOISE_RAYLEIGH    3
#define NOISE_ERLANG      4
#define NOISE_EXPONENTIAL 5
#define NOISE_PERIODIC    6

/* ================================================================
   IMAGE STRUCT
   ================================================================ */
typedef struct {
    unsigned char *im_prt;
    int width, height, channels;
} Image;

/* ----------------------------------------------------------------
   Internal helpers (static = not exported)
   ---------------------------------------------------------------- */

static Image *create_empty(int w, int h, int c) {
    Image *img  = (Image *)malloc(sizeof(Image));
    img->width  = w;
    img->height = h;
    img->channels = c;
    img->im_prt = (unsigned char *)calloc(w * h * c, 1);
    return img;
}

/* Clamp float -> uint8 */
static unsigned char clamp8(float v) {
    if (v < 0.0f)   return 0;
    if (v > 255.0f) return 255;
    return (unsigned char)v;
}

/* Min-max normalise float array -> uint8 dst */
static void normalise(float *src, unsigned char *dst, int n) {
    float mn = src[0], mx = src[0];
    for (int i = 1; i < n; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }
    if (mx == mn) { memset(dst, 128, n); return; }
    float range = mx - mn;
    for (int i = 0; i < n; i++)
        dst[i] = clamp8((src[i] - mn) / range * 255.0f);
}

/* qsort comparator for float */
static int cmp_f(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/*
 * Fill `win` with pixels from a (2*pad+1)^2 window around (r,c).
 * Clamps to image border. Returns element count.
 */
static int get_window(unsigned char *data, int row, int col,
                      int rows, int cols, int pad, float *win) {
    int cnt = 0;
    for (int dr = -pad; dr <= pad; dr++) {
        int rr = row + dr;
        if (rr < 0)    rr = 0;
        if (rr >= rows) rr = rows - 1;
        for (int dc = -pad; dc <= pad; dc++) {
            int cc = col + dc;
            if (cc < 0)    cc = 0;
            if (cc >= cols) cc = cols - 1;
            win[cnt++] = (float)data[rr * cols + cc];
        }
    }
    return cnt;
}

/* ================================================================
   EXPORTED: IMAGE I/O
   ================================================================ */

Image *readImage(const char *path, int desired_channels) {
    Image *img  = (Image *)malloc(sizeof(Image));
    img->im_prt = stbi_load(path, &img->width, &img->height,
                            &img->channels, desired_channels);
    if (desired_channels > 0) img->channels = desired_channels;
    if (!img->im_prt) { free(img); return NULL; }
    return img;
}

void writeImage(const char *path, Image *img, int quality) {
    if (img && img->im_prt)
        stbi_write_jpg(path, img->width, img->height,
                       img->channels, img->im_prt, quality);
}

void freeImage(Image *img) {
    if (img) {
        if (img->im_prt) free(img->im_prt);
        free(img);
    }
}

/* ================================================================
   EXPORTED: BASIC MATH OPS
   ================================================================ */

Image *add_images(Image *a, Image *b) {
    Image *out = create_empty(a->width, a->height, a->channels);
    int sz = a->width * a->height * a->channels;
    for (int i = 0; i < sz; i++) {
        int v = a->im_prt[i] + b->im_prt[i];
        out->im_prt[i] = v > 255 ? 255 : (unsigned char)v;
    }
    return out;
}

Image *sub_images(Image *a, Image *b) {
    Image *out = create_empty(a->width, a->height, a->channels);
    int sz = a->width * a->height * a->channels;
    for (int i = 0; i < sz; i++) {
        int v = a->im_prt[i] - b->im_prt[i];
        out->im_prt[i] = v < 0 ? 0 : (unsigned char)v;
    }
    return out;
}

Image *mul_images(Image *a, Image *b) {
    Image *out = create_empty(a->width, a->height, a->channels);
    int sz = a->width * a->height * a->channels;
    for (int i = 0; i < sz; i++) {
        int v = (a->im_prt[i] * b->im_prt[i]) / 255;
        out->im_prt[i] = v > 255 ? 255 : (unsigned char)v;
    }
    return out;
}

Image *div_images(Image *a, Image *b) {
    Image *out = create_empty(a->width, a->height, a->channels);
    int sz = a->width * a->height * a->channels;
    for (int i = 0; i < sz; i++) {
        if (b->im_prt[i] == 0) { out->im_prt[i] = 255; continue; }
        int v = (a->im_prt[i] * 255) / b->im_prt[i];
        out->im_prt[i] = v > 255 ? 255 : (unsigned char)v;
    }
    return out;
}

/* Generic convolution (multi-channel, skips border strip) */
Image *convolve(Image *in, float *kernel, int k_size) {
    Image *out = create_empty(in->width, in->height, in->channels);
    int off = k_size / 2;
    for (int y = off; y < in->height - off; y++) {
        for (int x = off; x < in->width - off; x++) {
            for (int c = 0; c < in->channels; c++) {
                float s = 0.0f;
                for (int ky = 0; ky < k_size; ky++)
                    for (int kx = 0; kx < k_size; kx++) {
                        int idx = ((y + ky - off) * in->width + (x + kx - off))
                                  * in->channels + c;
                        s += in->im_prt[idx] * kernel[ky * k_size + kx];
                    }
                int v = (int)s;
                out->im_prt[(y * in->width + x) * in->channels + c] =
                    v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
            }
        }
    }
    return out;
}

int *histogram(Image *in) {
    int *h = (int *)calloc(256, sizeof(int));
    int sz = in->width * in->height;
    for (int i = 0; i < sz; i++) h[in->im_prt[i * in->channels]]++;
    return h;
}

void freeHist(int *h) { free(h); }

/* ================================================================
   DCT — MATRIX METHOD
   Matrix entry: D[k,n] = alpha(k) * sqrt(2/N) * cos(pi*k*(2n+1)/(2N))
   2D-DCT(img) = D_M  *  img  *  D_N^T
   ================================================================ */

/*
 * Generate an N×N DCT-II matrix in ONE flat loop.
 * index = k*N + n  =>  k = index/N, n = index%N
 */
static float *make_dct_matrix(int N) {
    float *D   = (float *)malloc(N * N * sizeof(float));
    float scale = sqrtf(2.0f / (float)N);
    for (int idx = 0; idx < N * N; idx++) {
        int   k     = idx / N;
        int   n     = idx % N;
        float alpha = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
        D[idx] = alpha * scale
               * cosf((float)M_PI * k * (2.0f * n + 1.0f) / (2.0f * N));
    }
    return D;
}

/* C(M×N) = A(M×K) * B(K×N) */
static void mat_mul(float *A, float *B, float *C, int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
}

/* Return a new transposed copy of an N×N matrix */
static float *mat_transpose(float *A, int N) {
    float *T = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            T[i*N+j] = A[j*N+i];
    return T;
}

/*
 * 2-D DCT via separable matrix multiplication.
 * Input: grayscale Image*.  Output: single-channel normalised Image*.
 *
 * Steps:
 *   1. img_f = float version of image  (M×N)
 *   2. D_M   = M×M DCT matrix
 *   3. D_N   = N×N DCT matrix
 *   4. temp  = D_M * img_f             (M×N)
 *   5. coeff = temp * D_N^T            (M×N)
 *   6. normalise & pack into Image*
 */
Image *apply_dct(Image *in) {
    int M = in->height, N = in->width;

    float *img_f = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++)
        img_f[i] = (float)in->im_prt[i * in->channels];

    float *D_M   = make_dct_matrix(M);
    float *D_N   = make_dct_matrix(N);
    float *D_N_T = mat_transpose(D_N, N);

    float *temp  = (float *)malloc(M * N * sizeof(float));
    float *coeff = (float *)malloc(M * N * sizeof(float));

    mat_mul(D_M,  img_f, temp,  M, M, N);   /* D_M  * img_f  → temp  */
    mat_mul(temp, D_N_T, coeff, M, N, N);   /* temp * D_N^T  → coeff */

    Image *out = create_empty(N, M, 1);
    normalise(coeff, out->im_prt, M * N);

    free(img_f); free(D_M); free(D_N); free(D_N_T);
    free(temp);  free(coeff);
    return out;
}

/* ================================================================
   SPATIAL FILTERS — internal single-channel 2-D convolution
   ================================================================ */

static void conv2d_gray(unsigned char *data, int rows, int cols,
                        float *kernel, int ks, float *out) {
    int pad = ks / 2;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float s = 0.0f;
            for (int kr = 0; kr < ks; kr++) {
                int rr = r + kr - pad;
                if (rr < 0) rr = 0; else if (rr >= rows) rr = rows - 1;
                for (int kc = 0; kc < ks; kc++) {
                    int cc = c + kc - pad;
                    if (cc < 0) cc = 0; else if (cc >= cols) cc = cols - 1;
                    s += (float)data[rr * cols + cc] * kernel[kr * ks + kc];
                }
            }
            out[r * cols + c] = s;
        }
    }
}

/* ----------------------------------------------------------------
   MEAN FILTERS
   ---------------------------------------------------------------- */

Image *arithmetic_mean_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float s = 0; for (int i = 0; i < n; i++) s += win[i];
            of[r*cols+c] = s / n;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *geometric_mean_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float ls = 0;
            for (int i = 0; i < n; i++) ls += logf(win[i] < 1e-5f ? 1e-5f : win[i]);
            of[r*cols+c] = expf(ls / n);
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *harmonic_mean_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float rs = 0;
            for (int i = 0; i < n; i++) rs += 1.0f / (win[i] + 1e-5f);
            of[r*cols+c] = rs > 0 ? (float)n / rs : 0.0f;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *contraharmonic_mean_filter(Image *in, int fs, float Q) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float num = 0, den = 0;
            for (int i = 0; i < n; i++) {
                num += powf(win[i], Q + 1.0f);
                den += powf(win[i], Q);
            }
            of[r*cols+c] = den > 1e-5f ? num / den : 0.0f;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

/* ----------------------------------------------------------------
   ORDER-STATISTIC FILTERS
   ---------------------------------------------------------------- */

Image *median_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    int wsz = fs * fs;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(wsz * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            qsort(win, n, sizeof(float), cmp_f);
            of[r*cols+c] = win[n / 2];
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *max_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float mx = win[0];
            for (int i = 1; i < n; i++) if (win[i] > mx) mx = win[i];
            of[r*cols+c] = mx;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *min_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float mn = win[0];
            for (int i = 1; i < n; i++) if (win[i] < mn) mn = win[i];
            of[r*cols+c] = mn;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

Image *midpoint_filter(Image *in, int fs) {
    int rows = in->height, cols = in->width, pad = fs / 2;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(fs * fs * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            float mn = win[0], mx = win[0];
            for (int i = 1; i < n; i++) {
                if (win[i] < mn) mn = win[i];
                if (win[i] > mx) mx = win[i];
            }
            of[r*cols+c] = (mn + mx) * 0.5f;
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

/* d must be even */
Image *alpha_trimmed_filter(Image *in, int fs, int d) {
    if (d % 2 != 0) d--;
    int rows = in->height, cols = in->width, pad = fs / 2;
    int wsz = fs * fs;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(wsz * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int n = get_window(in->im_prt, r, c, rows, cols, pad, win);
            qsort(win, n, sizeof(float), cmp_f);
            int eff_d = d < (n - 1) ? d : (n - 1);
            if (eff_d % 2) eff_d--;
            int trim = eff_d / 2;
            float s = 0; int cnt = 0;
            for (int i = trim; i < n - trim; i++) { s += win[i]; cnt++; }
            of[r*cols+c] = cnt > 0 ? s / cnt : win[n / 2];
        }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows*cols);
    free(of); free(win); return out;
}

/*
 * Adaptive median filter.
 * Starts at filter_size, expands by 2 until median is not an impulse,
 * or max_filter_size is hit.
 */
Image *adaptive_median_filter(Image *in, int fs, int max_fs) {
    if (fs     % 2 == 0) fs++;
    if (max_fs % 2 == 0) max_fs++;
    int rows = in->height, cols = in->width;
    int max_wsz = max_fs * max_fs;
    float *of  = (float *)malloc(rows * cols * sizeof(float));
    float *win = (float *)malloc(max_wsz * sizeof(float));
    float *srt = (float *)malloc(max_wsz * sizeof(float));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int cur = fs, found = 0;
            float result = (float)in->im_prt[r * cols + c];

            while (cur <= max_fs) {
                int n = get_window(in->im_prt, r, c, rows, cols, cur / 2, win);
                memcpy(srt, win, n * sizeof(float));
                qsort(srt, n, sizeof(float), cmp_f);
                float mn  = srt[0], mx = srt[n-1], med = srt[n / 2];
                float pix = (float)in->im_prt[r * cols + c];

                if (mn < med && med < mx) {
                    /* median is not an impulse — check centre pixel */
                    result = (mn < pix && pix < mx) ? pix : med;
                    found  = 1;
                    break;
                }
                cur += 2;   /* window too small / all-impulse: grow */
            }

            if (!found) {
                /* max window reached — just use median of largest window */
                int n = get_window(in->im_prt, r, c, rows, cols, max_fs / 2, win);
                memcpy(srt, win, n * sizeof(float));
                qsort(srt, n, sizeof(float), cmp_f);
                result = srt[n / 2];
            }
            of[r * cols + c] = result;
        }
    }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows * cols);
    free(of); free(win); free(srt);
    return out;
}

/* ----------------------------------------------------------------
   SHARPENING FILTERS
   ---------------------------------------------------------------- */

/* Laplacian composite (centre-weighted kernel: keeps edges + brightness) */
Image *laplacian_sharpen_filter(Image *in) {
    float k[9] = {0,-1,0, -1,5,-1, 0,-1,0};
    int rows = in->height, cols = in->width;
    float *of = (float *)malloc(rows * cols * sizeof(float));
    conv2d_gray(in->im_prt, rows, cols, k, 3, of);
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, rows * cols);
    free(of); return out;
}

Image *sobel_filter(Image *in) {
    float kx[9] = {-1,0,1, -2,0,2, -1,0,1};
    float ky[9] = { 1,2,1,  0,0,0, -1,-2,-1};
    int rows = in->height, cols = in->width, n = rows * cols;
    float *gx = (float *)malloc(n * sizeof(float));
    float *gy = (float *)malloc(n * sizeof(float));
    float *of = (float *)malloc(n * sizeof(float));
    conv2d_gray(in->im_prt, rows, cols, kx, 3, gx);
    conv2d_gray(in->im_prt, rows, cols, ky, 3, gy);
    for (int i = 0; i < n; i++) of[i] = sqrtf(gx[i]*gx[i] + gy[i]*gy[i]);
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, n);
    free(gx); free(gy); free(of); return out;
}

Image *prewitt_filter(Image *in) {
    float kx[9] = {-1,0,1, -1,0,1, -1,0,1};
    float ky[9] = { 1,1,1,  0,0,0, -1,-1,-1};
    int rows = in->height, cols = in->width, n = rows * cols;
    float *gx = (float *)malloc(n * sizeof(float));
    float *gy = (float *)malloc(n * sizeof(float));
    float *of = (float *)malloc(n * sizeof(float));
    conv2d_gray(in->im_prt, rows, cols, kx, 3, gx);
    conv2d_gray(in->im_prt, rows, cols, ky, 3, gy);
    for (int i = 0; i < n; i++) of[i] = sqrtf(gx[i]*gx[i] + gy[i]*gy[i]);
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, n);
    free(gx); free(gy); free(of); return out;
}

/* alpha controls sharpening strength (1.0–2.0 typical) */
Image *unsharp_masking_filter(Image *in, int ks, float alpha) {
    int rows = in->height, cols = in->width, n = rows * cols;
    int kn = ks * ks;
    float *k = (float *)malloc(kn * sizeof(float));
    for (int i = 0; i < kn; i++) k[i] = 1.0f / (float)kn;

    float *blur = (float *)malloc(n * sizeof(float));
    float *of   = (float *)malloc(n * sizeof(float));
    conv2d_gray(in->im_prt, rows, cols, k, ks, blur);
    for (int i = 0; i < n; i++) {
        float p = (float)in->im_prt[i];
        of[i] = p + alpha * (p - blur[i]);   /* sharpen = img + alpha*mask */
    }
    Image *out = create_empty(cols, rows, 1);
    normalise(of, out->im_prt, n);
    free(k); free(blur); free(of); return out;
}

/* ================================================================
   FFT — Cooley-Tukey, radix-2, iterative, in-place
   ================================================================ */

typedef struct { float r, i; } Cx;

/* Round up to next power of 2 */
static int next_pow2(int n) {
    int p = 1; while (p < n) p <<= 1; return p;
}

/*
 * 1-D in-place FFT.  n must be a power of 2.
 * inverse == 0 → forward,  inverse == 1 → inverse (divides by n).
 */
static void fft1d(Cx *x, int n, int inverse) {
    /* bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { Cx t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    /* butterfly stages */
    for (int len = 2; len <= n; len <<= 1) {
        float ang  = 2.0f * (float)M_PI / (float)len * (inverse ? 1.0f : -1.0f);
        Cx wlen    = { cosf(ang), sinf(ang) };
        for (int i = 0; i < n; i += len) {
            Cx w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                Cx u = x[i + j];
                Cx v = { x[i+j+len/2].r * w.r - x[i+j+len/2].i * w.i,
                         x[i+j+len/2].r * w.i + x[i+j+len/2].i * w.r };
                x[i + j]          = (Cx){ u.r + v.r, u.i + v.i };
                x[i + j + len/2]  = (Cx){ u.r - v.r, u.i - v.i };
                w = (Cx){ w.r * wlen.r - w.i * wlen.i,
                          w.r * wlen.i + w.i * wlen.r };
            }
        }
    }
    if (inverse)
        for (int i = 0; i < n; i++) { x[i].r /= (float)n; x[i].i /= (float)n; }
}

/* 2-D FFT: FFT every row, then every column */
static void fft2d(Cx *data, int rows, int cols, int inverse) {
    for (int r = 0; r < rows; r++)
        fft1d(data + r * cols, cols, inverse);

    Cx *col = (Cx *)malloc(rows * sizeof(Cx));
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) col[r] = data[r * cols + c];
        fft1d(col, rows, inverse);
        for (int r = 0; r < rows; r++) data[r * cols + c] = col[r];
    }
    free(col);
}

/*
 * fftshift: move DC component to centre of the array.
 * For even sizes, applying twice restores original (works as ifftshift too).
 */
static void fftshift(Cx *data, int rows, int cols) {
    int hr = rows / 2, hc = cols / 2;
    Cx *tmp = (Cx *)malloc(rows * cols * sizeof(Cx));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            tmp[((r + hr) % rows) * cols + (c + hc) % cols] = data[r * cols + c];
    memcpy(data, tmp, rows * cols * sizeof(Cx));
    free(tmp);
}

/* ================================================================
   FREQUENCY FILTERS — shared core + per-filter mask builders
   ================================================================ */

/*
 * Core: forward FFT → shift → apply float mask → ishift → inverse FFT → normalise.
 * mask dimensions must be pr × pc (padded sizes).
 * Original-image region [0..rows-1][0..cols-1] is extracted after IFFT.
 */
static Image *freq_core(Image *in, float *mask, int pr, int pc) {
    int rows = in->height, cols = in->width;
    Cx *F = (Cx *)calloc(pr * pc, sizeof(Cx));

    /* copy image into real part (zero-padded) */
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            F[r * pc + c].r = (float)in->im_prt[r * cols + c];

    fft2d(F, pr, pc, 0);
    fftshift(F, pr, pc);

    for (int i = 0; i < pr * pc; i++) {
        F[i].r *= mask[i];
        F[i].i *= mask[i];
    }

    fftshift(F, pr, pc);    /* ifftshift == fftshift for even sizes */
    fft2d(F, pr, pc, 1);

    /* extract original-size real part */
    float *real = (float *)malloc(rows * cols * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            real[r * cols + c] = fabsf(F[r * pc + c].r);

    Image *out = create_empty(cols, rows, 1);
    normalise(real, out->im_prt, rows * cols);
    free(F); free(real);
    return out;
}

/* Euclidean distance from padded-array centre after fftshift */
static float D(int u, int v, int pr, int pc) {
    float du = (float)(u - pr / 2);
    float dv = (float)(v - pc / 2);
    return sqrtf(du * du + dv * dv);
}

/* -------- Low-pass -------- */

Image *ideal_low_pass_filter(Image *in, float cutoff) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)calloc(pr * pc, sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++)
            mask[u*pc+v] = D(u,v,pr,pc) <= cutoff ? 1.0f : 0.0f;
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *butterworth_low_pass_filter(Image *in, float cutoff, int order) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++)
            mask[u*pc+v] = 1.0f / (1.0f + powf(D(u,v,pr,pc) / cutoff,
                                                2.0f * (float)order));
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *gaussian_low_pass_filter(Image *in, float cutoff) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    float c2 = 2.0f * cutoff * cutoff;
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float d = D(u,v,pr,pc);
            mask[u*pc+v] = expf(-d*d / c2);
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

/* -------- High-pass -------- */

Image *ideal_high_pass_filter(Image *in, float cutoff) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++)
            mask[u*pc+v] = D(u,v,pr,pc) > cutoff ? 1.0f : 0.0f;
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *butterworth_high_pass_filter(Image *in, float cutoff, int order) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float d = D(u,v,pr,pc);
            mask[u*pc+v] = d < 1e-6f ? 0.0f
                         : 1.0f / (1.0f + powf(cutoff / d, 2.0f * (float)order));
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *gaussian_high_pass_filter(Image *in, float cutoff) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    float c2 = 2.0f * cutoff * cutoff;
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float d = D(u,v,pr,pc);
            mask[u*pc+v] = 1.0f - expf(-d*d / c2);
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

/* -------- Band-pass / Band-reject -------- */

Image *gaussian_band_pass_filter(Image *in, float low_cut, float high_cut) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float d  = D(u,v,pr,pc);
            float hi = expf(-d*d / (2.0f * high_cut * high_cut));
            float lo = expf(-d*d / (2.0f * low_cut  * low_cut));
            float m  = hi - lo;
            mask[u*pc+v] = m < 0.0f ? 0.0f : m;
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *gaussian_band_reject_filter(Image *in, float low_cut, float high_cut) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float d  = D(u,v,pr,pc);
            float hi = expf(-d*d / (2.0f * high_cut * high_cut));
            float lo = expf(-d*d / (2.0f * low_cut  * low_cut));
            float bp = hi - lo;
            if (bp < 0.0f) bp = 0.0f;
            mask[u*pc+v] = 1.0f - bp;
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

/* -------- Notch filters --------
 * notch_centers: flat array [u0,v0, u1,v1, ...] in SHIFTED coordinates.
 * n_centers    : number of (u,v) pairs.
 */

Image *gaussian_notch_reject_filter(Image *in, float *notch_centers,
                                    int n_centers, float radius) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    float r2 = 2.0f * radius * radius;
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float rej = 0.0f;
            for (int k = 0; k < n_centers; k++) {
                float du = u - notch_centers[k*2];
                float dv = v - notch_centers[k*2+1];
                rej += expf(-(du*du + dv*dv) / r2);
            }
            float m = 1.0f - rej;
            mask[u*pc+v] = m < 0.0f ? 0.0f : m;
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}

Image *gaussian_notch_pass_filter(Image *in, float *notch_centers,
                                  int n_centers, float radius) {
    int pr = next_pow2(in->height), pc = next_pow2(in->width);
    float *mask = (float *)malloc(pr * pc * sizeof(float));
    float r2 = 2.0f * radius * radius;
    for (int u = 0; u < pr; u++)
        for (int v = 0; v < pc; v++) {
            float pass = 0.0f;
            for (int k = 0; k < n_centers; k++) {
                float du = u - notch_centers[k*2];
                float dv = v - notch_centers[k*2+1];
                pass += expf(-(du*du + dv*dv) / r2);
            }
            mask[u*pc+v] = pass > 1.0f ? 1.0f : pass;
        }
    Image *out = freq_core(in, mask, pr, pc);
    free(mask); return out;
}



Image *auto_filter(Image *in, int noise_type) {
    switch (noise_type) {
        case NOISE_GAUSSIAN:
            return gaussian_low_pass_filter(in, 10.0f);
        case NOISE_SALT_PEPPER:
            return adaptive_median_filter(in, 3, 11);
        case NOISE_UNIFORM:
            return arithmetic_mean_filter(in, 5);
        case NOISE_RAYLEIGH:
            return geometric_mean_filter(in, 5);
        case NOISE_ERLANG:
            return geometric_mean_filter(in, 5);
        case NOISE_EXPONENTIAL:
            return harmonic_mean_filter(in, 5);
        case NOISE_PERIODIC:
            return gaussian_band_reject_filter(in, 20.0f, 40.0f);
        default:
            return median_filter(in, 5);
    }
}

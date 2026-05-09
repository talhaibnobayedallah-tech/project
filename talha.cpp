#include <iostream>
#include <cmath>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
    typedef struct {
        unsigned char* im_prt;
        int width;
        int height;
        int channels;
    } Image;

    // --- Memory & IO Helpers ---
    
    Image* createEmptyImage(int w, int h, int c) {
        Image* img = (Image*)malloc(sizeof(Image));
        img->width = w;
        img->height = h;
        img->channels = c;
        img->im_prt = (unsigned char*)calloc(w * h * c, sizeof(unsigned char));
        return img;
    }

    Image* readImage(const char* name_of_image, int desired_channels) {
        Image* nw_image = (Image*)malloc(sizeof(Image));
        nw_image->im_prt = stbi_load(name_of_image, &nw_image->width, &nw_image->height, &nw_image->channels, desired_channels);
        
        if (desired_channels > 0) nw_image->channels = desired_channels;

        if (nw_image->im_prt == nullptr) {
            cout << "Failure reason: " << stbi_failure_reason() << endl;
            free(nw_image);
            return nullptr;
        }
        return nw_image;
    }

    void writeImage(const char* name_to_save, Image* im_ptr, int quality) {
        if (im_ptr != nullptr && im_ptr->im_prt != nullptr) {
            // Using im_ptr->channels instead of hardcoded 1
            stbi_write_jpg(name_to_save, im_ptr->width, im_ptr->height, im_ptr->channels, im_ptr->im_prt, quality);
        }
    }

    void freeImage(Image* im_ptr) {
        if (im_ptr != nullptr) {
            // stbi_load uses malloc, so free() is safe here as well as for our custom createEmptyImage
            if (im_ptr->im_prt) free(im_ptr->im_prt);
            free(im_ptr);
        }
    }

    // --- Math Operators (+, -, *, /) ---

    Image* add_images(Image* a, Image* b) {
        Image* out = createEmptyImage(a->width, a->height, a->channels);
        int size = a->width * a->height * a->channels;
        for (int i = 0; i < size; i++) {
            int val = a->im_prt[i] + b->im_prt[i];
            out->im_prt[i] = (val > 255) ? 255 : val;
        }
        return out;
    }

    Image* sub_images(Image* a, Image* b) {
        Image* out = createEmptyImage(a->width, a->height, a->channels);
        int size = a->width * a->height * a->channels;
        for (int i = 0; i < size; i++) {
            int val = a->im_prt[i] - b->im_prt[i];
            out->im_prt[i] = (val < 0) ? 0 : val;
        }
        return out;
    }

    Image* mul_images(Image* a, Image* b) {
        Image* out = createEmptyImage(a->width, a->height, a->channels);
        int size = a->width * a->height * a->channels;
        for (int i = 0; i < size; i++) {
            int val = (a->im_prt[i] * b->im_prt[i]) / 255; // Normalize to avoid blowout
            out->im_prt[i] = (val > 255) ? 255 : val;
        }
        return out;
    }

    Image* div_images(Image* a, Image* b) {
        Image* out = createEmptyImage(a->width, a->height, a->channels);
        int size = a->width * a->height * a->channels;
        for (int i = 0; i < size; i++) {
            if (b->im_prt[i] == 0) out->im_prt[i] = 255; // Avoid division by zero
            else {
                int val = (a->im_prt[i] * 255) / b->im_prt[i];
                out->im_prt[i] = (val > 255) ? 255 : val;
            }
        }
        return out;
    }

    // --- Filters & Transforms ---

    Image* convolve(Image* in, float* kernel, int k_size) {
        Image* out = createEmptyImage(in->width, in->height, in->channels);
        int offset = k_size / 2;

        for (int y = offset; y < in->height - offset; y++) {
            for (int x = offset; x < in->width - offset; x++) {
                for (int c = 0; c < in->channels; c++) {
                    float sum = 0.0f;
                    // Apply kernel
                    for (int ky = 0; ky < k_size; ky++) {
                        for (int kx = 0; kx < k_size; kx++) {
                            int px = x + kx - offset;
                            int py = y + ky - offset;
                            int p_idx = (py * in->width + px) * in->channels + c;
                            int k_idx = ky * k_size + kx;
                            sum += in->im_prt[p_idx] * kernel[k_idx];
                        }
                    }
                    // Clamp
                    int v = (int)sum;
                    if (v < 0) v = 0;
                    if (v > 255) v = 255;
                    out->im_prt[(y * in->width + x) * in->channels + c] = v;
                }
            }
        }
        return out;
    }

    int* histogram(Image* in) {
        int* hist = (int*)calloc(256, sizeof(int));
        int size = in->width * in->height;
        // Basic: Using only the first channel for the histogram
        for (int i = 0; i < size; i++) {
            hist[in->im_prt[i * in->channels]]++;
        }
        return hist;
    }

    void freeHist(int* hist) {
        free(hist);
    }

    Image* apply_dct(Image* in) {
        // Warning: Very slow basic forward approach O(M^2 * N^2)
        Image* out = createEmptyImage(in->width, in->height, 1); 
        int M = in->height;
        int N = in->width;

        for (int u = 0; u < M; u++) {
            for (int v = 0; v < N; v++) {
                float sum = 0.0;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        int pixel = in->im_prt[(i * N + j) * in->channels]; // Use channel 0
                        sum += pixel * cos((M_PI * u * (2.0 * i + 1)) / (2.0 * M))
                                     * cos((M_PI * v * (2.0 * j + 1)) / (2.0 * N));
                    }
                }
                float cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
                float cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
                float val = 0.25 * cu * cv * sum;
                
                int p = abs((int)val);
                if (p > 255) p = 255;
                out->im_prt[u * N + v] = p;
            }
        }
        return out;
    }
}

#include <cstddef>
#include <cstdint>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;

struct {
unsigned char* im_prt;
int width;
int height;
int channels;
}typedef Image;


extern"C"{
Image readImage(const char* name_of_image, int desired_channels){
	Image nw_image;

	nw_image.im_prt = stbi_load(name_of_image, &nw_image.width, &nw_image.height, &nw_image.channels, desired_channels);
	if (nw_image.im_prt == NULL) {
		cout<<"can not find this image try again\n";
	}
	else {
		cout<<"Image is ready\n";

	}
	return nw_image;
}

}

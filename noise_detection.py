import ctypes as c

image = c.CDLL("./readImage.so")
image.readImage(b"/home/mostafa/programming/python/image_processing/project/1.jpg",0)

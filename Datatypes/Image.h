

#ifndef SCI_DATATYPES_IMAGE_H
#define SCI_DATATYPES_IMAGE_H 1

#include <Classlib/Array2.h>
#include <Geom/Color.h>

class Image {
public:
    Image(int xres, int yres);
    ~Image();
    Array2<Color> imagedata;
    inline Color& get_pixel(int x, int y) {
	return imagedata(y,x);
    }
    inline void put_pixel(int x, int y, const Color& pixel) {
	imagedata(y,x)=pixel;
    }
    int xres() const;
    int yres() const;
};

class DepthImage {
public:
    DepthImage(int xres, int yres);
    ~DepthImage();
    Array2<double> depthdata;
    double get_depth(int x, int y) {
	return depthdata(y,x);
    }
    inline void put_pixel(int x, int y, double depth) {
	depthdata(y,x)=depth;
    }
    int xres() const;
    int yres() const;
};

#endif

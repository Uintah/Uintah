
#include <Datatypes/Image.h>

Image::Image(int xres, int yres)
: imagedata(yres, xres)
{
}

Image::~Image()
{
}

DepthImage::DepthImage(int xres, int yres)
: depthdata(yres, xres)
{
}

DepthImage::~DepthImage()
{
}




#ifndef IMAGE_H
#define IMAGE_H 1

#include <Packages/rtrt/Core/Color.h>

#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
    inline Pixel(unsigned char r, unsigned char g, unsigned char b,
		 unsigned char a) : r(r), g(g), b(b), a(a)
    {
    }
    inline Pixel()
    {
    }
    inline void set(const Color& color) {
	float rr=color.r;
	rr=rr<0?0:rr>1?1:rr;
	float gg=color.g;
	gg=gg<0?0:gg>1?1:gg;
	float bb=color.b;
	bb=bb<0?0:bb>1?1:bb;
	r=(unsigned char)(rr*255.);
	g=(unsigned char)(gg*255.);
	b=(unsigned char)(bb*255.);
	a=255;
    }
};

class Image {
    char* buf;
    Pixel** image;
    int xres, yres;
    bool stereo;
public:
    Image(int xres, int yres, bool stereo);
    ~Image();
    void resize_image();
    inline int get_xres() const {
	return xres;
    }
    inline int get_yres() const {
	return yres;
    }
    inline bool get_stereo() const {
	return stereo;
    }
    void draw();
    void set(const Pixel& value);
    inline Pixel& operator()(int x, int y) {
	return image[y][x];
    }
    void save(char* file);
};
#if 0
class ImageTile;

class ViewTile {
  int lx,ly,ux,uy; // bounding box in screen space

  int whichImg; // currently used image
  
  int lastImg;  // last image referenced

  Array1< ImgTile > imgs; // array of image tiles to use

  // below are for volume rendering

  double stepDist;  // distance for each step - per tile variable...
  double stepAlpha; // per step alpha...

public:

  friend class ImageTile;
  
  void Draw(); // draws the correct child for this tile...
}; 

struct RayBundle;

class ImageTile {
  Image *img;
  double xo,yo;  // x/y origin on uv plane for tile
  
  double xs,ys;  // vectors can be used for mapping

  float  glscale; // for glPixelZoom

  int tileID;    // fixed number of potential tile sizes - 4x4 -> ??

  ViewTile *owner; // view tile that owns this

  Array1< RayBundle > bundles; // bundles for given rep
};

struct SampleBundle {
  Array1<int> xi;
  Array1<int> yi; // local frame...
};

// these are what is encoded in the PQ...

struct RayBundle {
  int RayPattern;   // fixed patterns exist for any portion of a tile

  // above could be a piece of local memmory if you are worried about
  // shared reads...

  ImageTile *owner; // tile that owns this bundle - to get offsets etc.

  double time;      // time when bundle was inserted
  
  double priority;  // priority of this bundle...

};
#endif

} // end namespace rtrt

#endif

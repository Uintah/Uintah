

#ifndef IMAGE_H
#define IMAGE_H 1

#include <sci_defs.h> // For HAVE_OOGL
#if defined(HAVE_OOGL)
#undef HAVE_OOGL
#endif

#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

#if defined(HAVE_OOGL)

///////////////////////////////////////////
// OOGL stuff
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1682
#pragma set woff 3201
#pragma set woff 3303
#pragma set woff 1506
#endif
#include <oogl/basicTexture.h>
#include <oogl/shadedPrim.h>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1682
#pragma reset woff 3201
#pragma reset woff 3303
#pragma reset woff 1506
#endif

class Blend : public GenAttrib
{
public:
  Blend() {}
  Blend( const Vec4f& color ) {m_color = color;}
  void alpha( float alpha ) {m_color[3] = alpha;}
  
protected:
  virtual void bindDef() {
    glEnable(GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glColor4fv( m_color.v() );
  }
  virtual void releaseDef() {
    glDisable(GL_BLEND);
  }

private:
  Vec4f m_color;
};
//
///////////////////////////////////////////

#else

class BasicTexture {};
class ShadedPrim {};
class Blend {};

#endif // HAVE_OOGL

namespace rtrt {
  class Image;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Image*&);
}

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

  unsigned char& operator[](int idx) {
    return (&r)[idx];
  }

  inline void set(const Color& color) {
    float rr=color.r;
    rr=rr<0.f?0.f:rr>1.f?1.f:rr;
    float gg=color.g;
    gg=gg<0.f?0.f:gg>1.f?1.f:gg;
    float bb=color.b;
    bb=bb<0.f?0.f:bb>1.f?1.f:bb;
    r=(unsigned char)(rr*255.f);
    g=(unsigned char)(gg*255.f);
    b=(unsigned char)(bb*255.f);
    a=255;
  }
};

class Image : public SCIRun::Persistent {
  char* buf;
  Pixel** image;
  int xres, yres;
  bool stereo;
public:
  Image(int xres, int yres, bool stereo);
  ~Image();
  Image() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Image*&);

  void resize_image();
  void resize_image(const int new_xres, const int new_yres);
  
  inline int get_xres() const { return xres; }
  inline int get_yres() const { return yres; }
  inline bool get_stereo() const { return stereo; }

  void draw( int  window_size, // 0 == large, 1 == medium
	     bool fullscreen ); 

  void set(const Pixel& value);
  inline Pixel& operator()(int x, int y) {
    return image[y][x];
  }
  void set(int x, int y, const Color& value) {
    image[y][x].set(value);
  }
  void save(char* file);
  void save_ppm(char *filename);
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

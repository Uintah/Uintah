
#ifndef SHAREDTEXTURE_H
#define SHAREDTEXTURE_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array2.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace std;

namespace rtrt {
  class SharedTexture;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::SharedTexture*&);
}

namespace rtrt { 
  class SharedTexture : public Material {
  public:
    enum Mode {
      Tile,
      Clamp,
      Nothing
    };
    
  protected:
    Mode umode, vmode;
    Array2<Color> image;
    //this is for the alpha values
    Array2<float> alpha;
    
    Color outcolor;
    bool valid_;
    string filename_;
    
  public:
    SharedTexture(const string &filename, Mode umode=Clamp, Mode vmode=Clamp,
		  bool flipped=false);
    virtual ~SharedTexture();
    
    SharedTexture() : Material() {} // for Pio.
    
    //! Persistent I/O.
    static  SCIRun::PersistentTypeID type_id;
    virtual void io(SCIRun::Piostream &stream);
    friend void SCIRun::Pio(SCIRun::Piostream&, SharedTexture*&);
    
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth, 
		       double atten, const Color& accumcolor,
		       Context* cx);
    bool valid() { return valid_; }
    Color interp_color(Array2<Color>& image, double u, double v);
    float return_alpha(Array2<float>& alpha, double u, double v);
  };
} // end namespace rtrt

#endif

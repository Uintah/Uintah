
#ifndef TILEIMAGEMATERIAL_H
#define TILEIMAGEMATERIAL_H 1

#include <Packages/rtrt/Core/ImageMaterial.h>
#include <string>

namespace rtrt {
class TileImageMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::TileImageMaterial*&);
}

namespace rtrt { 

using std::string;

class TileImageMaterial : public ImageMaterial {

public:
  TileImageMaterial(int /* oldstyle */, const string &filename, 
		    double Kd, const Color& specular,
		    double specpow, double refl=0, bool flipped=0);
  TileImageMaterial(const string &filename,
		    double Kd, const Color& specular,
		    double specpow, double refl, 
		    double transp=0, bool flipped=false);
  TileImageMaterial(const string &filename,
		    double Kd, const Color& specular,
		    double specpow, double refl=0, bool flipped=false);
  virtual ~TileImageMaterial() {}

  TileImageMaterial() : ImageMaterial() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TileImageMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif

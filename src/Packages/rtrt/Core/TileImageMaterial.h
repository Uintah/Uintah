
#ifndef TILEIMAGEMATERIAL_H
#define TILEIMAGEMATERIAL_H 1

#include <Packages/rtrt/Core/ImageMaterial.h>
#include <string>

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
    virtual void shade(Color& result, const Ray& ray,
                       const HitInfo& hit, int depth, 
                       double atten, const Color& accumcolor,
                       Context* cx);
};

} // end namespace rtrt

#endif

#ifndef MIPMATERIAL_H
#define MIPMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/ImageMaterial.h>

#include <string>

using namespace std;
using rtrt::ImageMaterial;

namespace rtrt {
  class MIPMaterial;
}

/*  namespace SCIRun { */
/*    void Pio(Piostream&, rtrt::MIPMaterial*&); */
/*  } */

namespace rtrt {
class MIPMaterial : public Material
{
    
public:
    Color ambient;
    double Kd;
    Color specular;
    double specpow;
    double refl;
    string filename_;
    bool valid_;
    Color outcolor;
    Array2<Color> *image;
    int n_images;

    MIPMaterial() : Material() {} // for Pio.
    MIPMaterial(const string& filename,
                double Kd, const Color specular,
		double specpow, double refl=0,
		bool flipped=0);
    MIPMaterial(const string& filename, rtrt::ImageMaterial::Mode, 
                rtrt::ImageMaterial::Mode,
                double Kd, const Color specular,
		double specpow, double refl=0,
		bool flipped=0);

/*      static  SCIRun::PersistentTypeID type_id; */
/*      virtual void io(SCIRun::Piostream &stream); */
/*      friend void SCIRun::Pio(SCIRun::Piostream&, MIPMaterial*&); */

    ~MIPMaterial();
    virtual void shade(Color& result, const Ray& ray,
                       const HitInfo& hit, int depth, 
                       double atten, const Color& accumcolor,
                       Context* cx);
};
}
#endif

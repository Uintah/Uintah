
#ifndef IMAGEMATERIAL_H
#define IMAGEMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array2.h>
#include <string>

namespace rtrt { 

class ImageMaterial : public Material {
public:
    enum Mode {
	Tile,
	Clamp,
	None
    };
protected:

    Mode umode, vmode;
    double Kd;
    Color specular;
    double specpow;
    double refl;
    double transp;
    Array2<Color> image;
    Color outcolor;
    bool valid_;

    void read_image(const string &texfile);
    void read_hdr_image(const string &texfile);
public:
    ImageMaterial(int /* oldstyle */, const string &filename, 
		  Mode umode, Mode vmode,
		  double Kd, const Color& specular,
		  double specpow, double refl=0, bool flipped=0);
    ImageMaterial(const string &filename, Mode umode, Mode vmode,
		  double Kd, const Color& specular,
		  double specpow, double refl, 
		  double transp=0, bool flipped=false);
    ImageMaterial(const string &filename, Mode umode, Mode vmode,
		  double Kd, const Color& specular,
		  double specpow, double refl=0, bool flipped=false);
    virtual ~ImageMaterial();
    virtual void shade(Color& result, const Ray& ray,
                       const HitInfo& hit, int depth, 
                       double atten, const Color& accumcolor,
                       Context* cx);
    bool valid() { return valid_; }
    Color interp_color(Array2<Color>& image, double u, double v);
    void set_refl(double r)
      {
	refl = r;
      }
};

} // end namespace rtrt

#endif

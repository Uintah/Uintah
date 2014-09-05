
#ifndef IMAGEMATERIAL_H
#define IMAGEMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array2.h>

#include <string>

using namespace std;

namespace rtrt {
  class ImageMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::ImageMaterial*&);
}

namespace rtrt { 

class ImageMaterial : public Material {
public:
  enum Mode {
    Tile,
    Clamp,
    Nothing
  };
protected:

  Mode umode, vmode;
  double Kd;
  Color specular;
  double specpow;
  double refl;
  double transp;
  Array2<Color> image;
  //this is for the alpha values
  Array2<float> alpha;

  Color outcolor;
  bool valid_;
  string filename_;

  void read_hdr_image(const string &texfile);
public:
  ImageMaterial(int /* oldstyle */, const string &filename, 
		Mode umode, Mode vmode,
		double Kd, const Color& specular,
		double specpow, double refl=0, bool flipped=false);
  ImageMaterial(const string &filename, Mode umode, Mode vmode,
		double Kd, const Color& specular,
		double specpow, double refl=0, bool flipped=false);
  ImageMaterial(const string &filename, Mode umode, Mode vmode,
		const Color& specular, double Kd,
		double specpow, double refl, 
		double transp=0, bool flipped=false);
  virtual ~ImageMaterial();
    

  ImageMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, ImageMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
  bool valid() { return valid_; }
  Color interp_color(Array2<Color>& image, double u, double v);
  //float return_alpha(Array2<float>& alpha, double u, double v)
  void set_refl(double r)
  {
    refl = r;
  }
};

} // end namespace rtrt

#endif


#ifndef NORMALMAP_MAT_H
#define NORMALMAP_MAT_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/LambertianMaterial.h> 
#include <Packages/rtrt/Core/BumpObject.h> 

#include <fstream>
#include <iostream>

namespace rtrt {
class NormalMapMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::NormalMapMaterial*&);
}

namespace rtrt {

class NormalMapMaterial : public Material {
  Material *material;
  //added for bumpmaps from file
  int dimension_x;             /* width and height*/
  int dimension_y;
  Vector *normalmapimage;      /*holds the bump structure*/
  double evaru, evarv; /* 1/width, 1/height */
  double persistence;
public:
  FILE* readcomments(FILE *fin);
  int readfromppm6(char *filename);
  int readfromppm(char *filename);
  NormalMapMaterial(Material *, char *, double);
  virtual ~NormalMapMaterial();

  NormalMapMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, NormalMapMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  void perturbnormal(Vector &, const Ray &, const HitInfo &);

  // added for file bumpmaps
  int read_file(char *filename);
  Vector fval(double u, double v);
  double get_persistence() {return persistence;}

};

} // end namespace rtrt

#endif

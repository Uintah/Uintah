
#ifndef BUMP_MAT_H
#define BUMP_MAT_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/LambertianMaterial.h> 
#include <Packages/rtrt/Core/BumpObject.h> 

#include <fstream>
#include <iostream>

namespace rtrt {
class BumpMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::BumpMaterial*&);
}

namespace rtrt {

class BumpMaterial : public Material {
  Material *material;
  //added for bumpmaps from file
  int dimension_x;             /* width and height*/
  int dimension_y;
  int *bumpimage;      /*holds the bump structure*/
  double evaru, evarv; /* 1/width, 1/height */
  double ntiles;
  double bump_scale;
public:
  BumpMaterial(Material *, char *, double, double bump_scale=1);
  virtual ~BumpMaterial();

  BumpMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, BumpMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  void perturbnormal(Vector &, const Ray& ray, const HitInfo &);
  
  
  // added for file bumpmaps
  int read_file(char *filename);
  double fval(double u, double v);
  inline double get_ntiles() {return ntiles;}
  int readfromppm6(char *);
  //int readfromppm(char *);

  FILE * readcomments (FILE*fin);
  void writetoppm(char *filename,char *bumpfile, Vector *v);
  void CreateNormalMap(char *bumpfile, char *filename, Vector normal, Vector pu, Vector pv);
  void print();

};

} // end namespace rtrt

#endif

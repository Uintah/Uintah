
#ifndef BUMP_MAT_H
#define BUMP_MAT_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/LambertianMaterial.h> 
#include <Packages/rtrt/Core/BumpObject.h> 

#include <fstream>
#include <iostream>

namespace rtrt {

class BumpMaterial : public Material {
  Material *material;
  //added for bumpmaps from file
  int dimension_x;             /* width and height*/
  int dimension_y;
  int *bumpimage;      /*holds the bump structure*/
  double evaru, evarv; /* 1/width, 1/height */
  double persistence;
public:
    BumpMaterial(Material *, char *, double);
    virtual ~BumpMaterial();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
  void perturbnormal(Vector &, const Ray& ray, const HitInfo &);


  // added for file bumpmaps
  int read_file(char *filename);
  double fval(double u, double v);
  double get_persistence() {return persistence;}
  int readfromppm6(char *);
  //int readfromppm(char *);
  FILE * readcomments (FILE*fin);

};

} // end namespace rtrt

#endif

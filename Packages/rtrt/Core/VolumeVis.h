#ifndef __RTRT_VOLUMEVIS_H__
#define __RTRT_VOLUMEVIS_H__

#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <stdlib.h>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

class VolumeVisDpy;

class VolumeVis : public Object, public Material {
protected:
  friend class VolumeVisDpy;
  VolumeVisDpy *dpy;
  
  Vector diag;
  Vector inv_diag;
  BrickArray3<float> data;
  float data_min, data_max;
  int nx, ny, nz;
  Point min, max;
  double spec_coeff, ambient, diffuse, specular;
  float delta_x2, delta_y2, delta_z2;
  
  inline int clamp(const int min, const int val, const int max) {
    return (val>min?(val<max?val:max):min);
  }
  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color);
public:
  VolumeVis(BrickArray3<float>& data, float data_min, float data_max,
	    int nx, int ny, int nz, Point min, Point max,
	    double spec_coeff, double ambient,
	    double diffuse, double specular, VolumeVisDpy *dpy);
  virtual ~VolumeVis();
  //! Persistent I/O.
  //static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  //friend void SCIRun::Pio(SCIRun::Piostream&, VolumeVis*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  virtual void animate(double t, bool& changed);
};
  
} // end namespace rtrt

#endif

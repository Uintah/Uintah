#ifndef __RTRT_VOLUMEVIS2D_H__
#define __RTRT_VOLUMEVIS2D_H__

#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <stdlib.h>
#include <iostream>

namespace rtrt {

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

class VolumeVis2DDpy;

template<class T>
class Voxel2D {
public:
  T _data[2];
public:
  Voxel2D(const T& _v, const T& _g) {
    _data[0] = _v;
    _data[1] = _g;
  }
  Voxel2D(const T& _val) {
    _data[0] = _val;
    _data[1] = _val;
  }
  
  inline T& v() { return _data[0]; }
  inline T& g() { return _data[1]; }

  inline Voxel2D<T> operator+(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() + vox.v(), g() + vox.g());
  }
  inline Voxel2D<T> operator-(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() - vox.v(), g() - vox.g());
  }
  inline Voxel2D<T> operator*(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() * vox.v(), g() * vox.g());
  }

  friend std::ostream& operator<<(std::ostream& out, const Voxel2D<T>& c);
};

template<class T>
std::ostream& operator<<(std::ostream& out, const Voxel2D<T>& vox) {
  out << "(" << vox.v() << ", " << vox.g() << ")";
  return out;
}

  
template<class T>
inline Voxel2D<T> operator-(T& lhs, const Voxel2D<T>& rhs)
{
  return (Voxel2D<T>)lhs - rhs;
}

  
class VolumeVis2D : public Object, public Material {
protected:
  friend class VolumeVis2DDpy;
  VolumeVis2DDpy *dpy;
  
  Vector diag;
  Vector inv_diag;
  BrickArray3<Voxel2D<float>> data;
  float data_min, data_max;
  int nx, ny, nz;
  Point min, max;
  double spec_coeff, ambient, diffuse, specular;
  float delta_x2, delta_y2, delta_z2;
  
  inline int bound(const int val, const int min, const int max) {
    return (val>min?(val<max?val:max):min);
  }
  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color);
public:
  VolumeVis2D(BrickArray3<Voxel2D<float> >& data,
	      Voxel2D<float> data_min, Voxel2D<float> data_max,
	    int nx, int ny, int nz, Point min, Point max,
	    double spec_coeff, double ambient,
	    double diffuse, double specular, VolumeVis2DDpy *dpy);
  virtual ~VolumeVis2D();
  //! Persistent I/O.
  //static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  //friend void SCIRun::Pio(SCIRun::Piostream&, VolumeVis2D*&);

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

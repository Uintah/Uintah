#ifndef __RTRT_VOLUMEVIS2D_H__
#define __RTRT_VOLUMEVIS2D_H__

#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <stdlib.h>
#include <iostream>
#include <map>
#include <Packages/rtrt/Core/MouseCallBack.h>

namespace rtrt {

#define CLEAN 0
#define FAST 1

#ifndef MAXUNSIGNEDSHORT
#define MAXUNSIGNEDSHORT 65535
#endif

  // template<class T>
class VolumeVis2DDpy;

template<class T>
class Voxel2D {
public:
  T _data[2];
public:
  Voxel2D() {}
  Voxel2D(const T& _v, const T& _g) {
    _data[0] = _v;
    _data[1] = _g;
  }
  Voxel2D(const T& _val) {
    _data[0] = _val;
    _data[1] = _val;
  }
  
  inline T v() const { return _data[0]; }
  inline T g() const { return _data[1]; }
  inline T& vref() { return _data[0]; }
  inline T& gref() { return _data[1]; }

  inline Voxel2D<T> operator+(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() + vox.v(), g() + vox.g());
  }
  inline Voxel2D<T> operator-(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() - vox.v(), g() - vox.g());
  }
  inline Voxel2D<T> operator*(const Voxel2D<T>& vox) {
    return Voxel2D<T>(v() * vox.v(), g() * vox.g());
  }
  inline Voxel2D<T> operator*(T val) {
    return Voxel2D<T>(v() * val, g() * val);
  }

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

  
// template<class T>
class VolumeVis2D : public Object, public Material {
protected:
  friend class Volvis2DDpy;
  Volvis2DDpy *dpy;
  
  Vector diag;
  Vector inv_diag;
  BrickArray3<Voxel2D<float> > data;
  Voxel2D<float> data_min, data_max;
  int nx, ny, nz;
  float norm_step_x, norm_step_y, norm_step_z;
  Point min, max;
  double spec_coeff, ambient, diffuse, specular;
  float delta_x2, delta_y2, delta_z2;

  // stuff for cutting plane
  enum CPlane_overwrite { OverwroteTMax, OverwroteTMin, Neither };
  struct VolumeVis2D_scratchpad {
    CPlane_overwrite coe;
    float tmax;
  };
  Vector cutplane_normal;
  double cutplane_displacement;
  PlaneDpy* cdpy;
  bool cutplane_active;
  bool use_cutplane_material;
  
  // stuff for callback functions


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
	    double diffuse, double specular, Volvis2DDpy *dpy);
  virtual ~VolumeVis2D();
  //! Persistent I/O.
  //static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  //friend void SCIRun::Pio(SCIRun::Piostream&, VolumeVis2D*&);

  virtual void intersect( Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* ppc );
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  virtual void point2indexspace( Point &p,
				 int& xl, int& xh, float& xwl,
				 int& yl, int& yh, float& ywl,
				 int& zl, int& zh, float& zwl );
  virtual bool lookup_value( Point &p,
			     Voxel2D<float> &return_value,
			     bool exit_early );
  virtual bool lookup_value( Voxel2D<float> &return_value, bool exit_early,
			     int xl, int xh, float xwl,
			     int yl, int yh, float ywl,
			     int zl, int zh, float zwl );
  virtual void compute_grad( Ray r, Point p, Vector gradient, float &opacity,
			     Color value_color, Color &total, Context* cx );
  virtual void animate(double t, bool& changed);
  virtual void cblookup( Object* obj );
  virtual void initialize_cuttingPlane( PlaneDpy *cdpy );
  void initialize_callbacks() {
    MouseCallBack::assignCB_MD( VolumeVis2D::mouseDown_Wrap, this );
    MouseCallBack::assignCB_MU( VolumeVis2D::mouseUp_Wrap, this );
    MouseCallBack::assignCB_MM( VolumeVis2D::mouseMotion_Wrap, this );
  }
  static void mouseDown_Wrap( int x, int y, Object *obj,
			      const Ray& ray, const HitInfo& hit ) {
    VolumeVis2D *vobj = (VolumeVis2D*)obj;
    vobj->mouseDown( x, y, ray, hit );
  }
  void mouseDown( int x, int y, const Ray& ray, const HitInfo& hit );
  static void mouseUp_Wrap( int x, int y, Object *obj,
			    const Ray& ray, const HitInfo& hit) {
    VolumeVis2D *vobj = (VolumeVis2D*) obj;
    vobj->mouseUp( x, y, ray, hit );
  }
  void mouseUp( int x, int y, const Ray& ray, const HitInfo& hit );
  static void mouseMotion_Wrap( int x, int y, Object *obj,
				const Ray& ray, const HitInfo& hit ) {
    VolumeVis2D *vobj = (VolumeVis2D*) obj;
    vobj->mouseMotion( x, y, ray, hit );
  }
  void mouseMotion( int x, int y, const Ray& ray, const HitInfo& hit );
};

} // end namespace rtrt

#endif

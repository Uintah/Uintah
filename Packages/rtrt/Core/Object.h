

#ifndef OBJECT_H
#define OBJECT_H 1

#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Transform.h>
#include <Core/Persistent/Persistent.h>
#include <iostream>

#include <string>

namespace SCIRun {
  class Point;
  class Vector;
  class Transform;
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;
using std::string;

struct DepthStats;

class  HitInfo;
class  Material;
class  Ray;
class  Light;
class  BBox;
class  PerProcessorContext;
class  UVMapping;
class Object;

template<class T> class Array1;
}
namespace SCIRun {
void Pio(Piostream&, rtrt::Object*&);
}

namespace rtrt {

class Object : public virtual SCIRun::Persistent {
  Material* matl;
  UVMapping* uv;
protected:
  bool was_preprocessed;
public:
  string name_;

  Object(Material* matl, UVMapping* uv=0);
  virtual ~Object();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Object*&);

  inline Material  * get_matl() const { return matl; }
  inline void        set_matl(Material* new_matl) { matl=new_matl; }
  inline UVMapping * get_uvmapping() { return uv; }
  inline void        set_uvmapping(UVMapping* uv) { this->uv=uv; }

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*)=0;
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit)=0;
  //    virtual void get_frame(const Point &p, Vector &n, Vector &u, Vector &v);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox& bbox, double offset)=0;
  virtual void collect_prims(Array1<Object*>& prims);
  virtual void print(ostream& out);
  virtual void transform(Transform&) {}

  // This function should return TRUE when the point in question
  // (ray.v * t + ray.t0) can be mapped to a value by the object.
  // It returns FALSE otherwise.
  virtual bool interior_value( double& /*value*/, const Ray &/*ray*/,
			       const double /*t*/)
  { return false; }; 

};

} // end namespace rtrt

#endif


#ifndef SPHERE_H
#define SPHERE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {
class Sphere;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Sphere*&);
}

namespace rtrt {

class Sphere : public Object {
protected:
  Point cen;
  double radius;
public:

  /*    void* operator new(size_t);
	void operator delete(void*, size_t);*/
  Sphere(Material* matl, const Point& cen, double radius);
  virtual ~Sphere();

  Sphere() : Object(0) {} // for Pio.
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Sphere*&);
  
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
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
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void print(ostream& out);
  // Be careful using this, as if the sphere is in an efficiency structure
  // this won't work correctly.
  void updatePosition( const Point & pos );
};

} // end namespace rtrt

#endif

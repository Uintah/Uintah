
#ifndef TRI_H
#define TRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class Tri;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Tri*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;
using SCIRun::AffineCombination;

class Rect;

class Tri : public Object {
  Point p1, p2, p3;
  Vector vn1, vn2, vn3;
  Vector n;
  double d;
  Vector e1p, e2p, e3p;
  Vector e1, e2, e3;
  double e1l, e2l, e3l;
  bool bad;
public:
  inline bool isbad() {
    return bad;
  }
  Tri(Material* matl, const Point& p1, const Point& p2, const Point& p3);
  Tri(Material* matl, const Point& p1, const Point& p2, const Point& p3,
      const Vector& vn1, const Vector& vn2, const Vector& vn3,
      bool check_badness = true);
  virtual ~Tri();

  Tri() : Object(0) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Tri*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st,PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  inline Vector normal() { return n; }
  virtual void compute_bounds(BBox&, double offset);

  // returns a new rect that combines me and tri if we form a rect, else NULL
  Rect * pairMeUp( Tri * tri );

  Point centroid()
  {
    double one_third = 1./3.;

    return AffineCombination(p1,one_third,
			     p2,one_third,
			     p3,one_third);
  }
  Point pt(int i)
  {
    if (i==0)
      return p1;
    else if (i==1)
      return p2;
    else 
      return p3;
  }
	       
  inline Tri copy_transform(Transform& T)
  {

    Point tp1 = T.project(p1);
    Point tp2 = T.project(p2);
    Point tp3 = T.project(p3);
	  
    Vector tvn1 = T.project_normal(vn1);
    Vector tvn2 = T.project_normal(vn2);
    Vector tvn3 = T.project_normal(vn3);

    if (!isbad()) {
      tvn1.normalize();
      tvn2.normalize();
      tvn3.normalize();
    }

    return Tri(this->get_matl(),
	       tp1,tp2,tp3,
	       tvn1,tvn2,tvn3,false);
  }

  void transform(Transform& T)
  {
    *this = copy_transform(T);
  }
};

} // end namespace rtrt

#endif

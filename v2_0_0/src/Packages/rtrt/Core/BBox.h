
#ifndef BBOX_H
#define BBOX_H 1

#include <Packages/rtrt/Core/Ray.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>

namespace rtrt {
  class BBox;
}
namespace SCIRun {
  void Pio(Piostream&, rtrt::BBox*&);
}

namespace rtrt {

using SCIRun::Transform;
using SCIRun::Point;
using SCIRun::Min;
using SCIRun::Max;

class BBox : public SCIRun::Persistent {
protected:
  Point cmin;
  Point cmax;
  bool have_some;
public:
  inline BBox() {
    have_some=false;
  }
  inline BBox(const Point& min, const Point &max) : cmin(min), cmax(max), have_some(true) {
  }

  ~BBox() {
  }
  BBox(const BBox& copy);

  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::BBox*&);

  inline int valid() const {return have_some;}
  void reset();
  inline void extend(const Point& p){
    if(have_some){
      cmin=Min(cmin, p);
      cmax=Max(cmax, p);
    } else {
      cmin=cmax=p;
      have_some=true;
    }
  }
  inline void extend(double* slabs){
    extend(Point(slabs[0], slabs[1], slabs[2]));
    extend(Point(slabs[3], slabs[4], slabs[5]));
  }
  inline void extend(const Point& p, double radius) {
    if(!have_some){
      cmin=cmax=p;
      have_some=true;
    }
    cmin=Min(cmin, p-Vector(radius, radius, radius));
    cmax=Max(cmax, p+Vector(radius, radius, radius));
  }

  inline void extend(const BBox& b) {
    if(have_some){
      if(b.have_some){
	cmin=Min(cmin, b.cmin);
	cmax=Max(cmax, b.cmax);
      }
    } else {
      if(b.have_some){
	cmin=b.cmin;
	cmax=b.cmax;
	have_some=true;
      }
    }
  }

  inline void extend(double radius)
    {
      if (have_some) {
	cmin = cmin-Vector(radius,radius,radius);
	cmax = cmax+Vector(radius,radius,radius);
      }
    }

  bool intersect_nontrivial(const Ray& ray, double &min_t);

  inline bool intersect(const Ray& ray, double &min_t) {
    // if we're already inside the box, set min_t to zero and return true
    if (cmin.x()<ray.origin().x() && ray.origin().x()<cmax.x() &&
	cmin.y()<ray.origin().y() && ray.origin().y()<cmax.y() &&
	cmin.z()<ray.origin().z() && ray.origin().z()<cmax.z()) {
      min_t=0;
      return true;
    } else return intersect_nontrivial(ray, min_t);
  }

  inline bool contains_point(const Point &p) const {
    return (cmin.x()<=p.x() && p.x()<=cmax.x() &&
	    cmin.y()<=p.y() && p.y()<=cmax.y() &&
	    cmin.z()<=p.z() && p.z()<=cmax.z());
  }
  inline bool contains_point(const Ray &ray, double &t) const {
    return contains_point(ray.eval(t));
  }
  
  Point center() const;
  double longest_edge();
  inline Point min() const {
    return cmin;
  }
  inline Point max() const {
    return cmax;
  }
  inline Vector diagonal() const {
    return cmax-cmin;
  }
  inline BBox transform(Transform* t) const
  {
    BBox tbbox;

    //necessary for arbitrary rotations, can't just use min and max
    double x0 = cmin.x();
    double y0 = cmin.y();
    double z0 = cmin.z();
    double z1 = cmax.z();
    double y1 = cmax.y();
    double x1 = cmax.x();

    Point b000(x0,y0,z0);
    Point b001(x0,y0,z1);
    Point b010(x0,y1,z0);
    Point b011(x0,y1,z1);
    Point b100(x1,y0,z0);
    Point b101(x1,y0,z1);
    Point b110(x1,y1,z0);
    Point b111(x1,y1,z1);


    t->project_inplace(b000);
    t->project_inplace(b001);
    t->project_inplace(b010);
    t->project_inplace(b011);
    t->project_inplace(b100);
    t->project_inplace(b101);
    t->project_inplace(b110);
    t->project_inplace(b111);

    tbbox.have_some = false;
    tbbox.extend(b000);
    tbbox.extend(b001);
    tbbox.extend(b010);
    tbbox.extend(b011);
    tbbox.extend(b100);
    tbbox.extend(b101);
    tbbox.extend(b110);
    tbbox.extend(b111);
		
    return tbbox;
  }
    
  inline void transform(Transform* t, BBox& tbbox) const
  {
    //necessary for arbitrary rotations, can't just use min and max
    double x0 = cmin.x();
    double y0 = cmin.y();
    double z0 = cmin.z();
    double z1 = cmax.z();
    double y1 = cmax.y();
    double x1 = cmax.x();

    Point b000(x0,y0,z0);
    Point b001(x0,y0,z1);
    Point b010(x0,y1,z0);
    Point b011(x0,y1,z1);
    Point b100(x1,y0,z0);
    Point b101(x1,y0,z1);
    Point b110(x1,y1,z0);
    Point b111(x1,y1,z1);


    t->project_inplace(b000);
    t->project_inplace(b001);
    t->project_inplace(b010);
    t->project_inplace(b011);
    t->project_inplace(b100);
    t->project_inplace(b101);
    t->project_inplace(b110);
    t->project_inplace(b111);

    tbbox.have_some = false;
    tbbox.extend(b000);
    tbbox.extend(b001);
    tbbox.extend(b010);
    tbbox.extend(b011);
    tbbox.extend(b100);
    tbbox.extend(b101);
    tbbox.extend(b110);
    tbbox.extend(b111);
  }

  inline void transform_inplace(Transform* t)
  {
    double x0 = cmin.x();
    double y0 = cmin.y();
    double z0 = cmin.z();
    double z1 = cmax.z();
    double y1 = cmax.y();
    double x1 = cmax.x();

    Point b000(x0,y0,z0);
    Point b001(x0,y0,z1);
    Point b010(x0,y1,z0);
    Point b011(x0,y1,z1);
    Point b100(x1,y0,z0);
    Point b101(x1,y0,z1);
    Point b110(x1,y1,z0);
    Point b111(x1,y1,z1);


    t->project_inplace(b000);
    t->project_inplace(b001);
    t->project_inplace(b010);
    t->project_inplace(b011);
    t->project_inplace(b100);
    t->project_inplace(b101);
    t->project_inplace(b110);
    t->project_inplace(b111);

    have_some = false;
    extend(b000);
    extend(b001);
    extend(b010);
    extend(b011);
    extend(b100);
    extend(b101);
    extend(b110);
    extend(b111);
  }
    
};

} // end namespace rtrt

#endif


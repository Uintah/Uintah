
#ifndef BBOX_H
#define BBOX_H 1

#include <Packages/rtrt/Core/Ray.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>

namespace rtrt {

using SCIRun::Transform;
using SCIRun::Point;
using SCIRun::Min;
using SCIRun::Max;

class BBox {
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
	
	tbbox.cmin = t->project(cmin);
	tbbox.cmax = t->project(cmax);
	
	tbbox.have_some = have_some;
	
	return tbbox;
      }
    
    inline void transform(Transform* t, BBox& tbbox) const
      {
	t->project(cmin,tbbox.cmin);
	t->project(cmax,tbbox.cmax);
	
	tbbox.have_some = have_some;
      }
    inline void transform_inplace(Transform* t)
      {
	t->project_inplace(cmin);
	t->project_inplace(cmax);
      }
    
};

} // end namespace rtrt

#endif


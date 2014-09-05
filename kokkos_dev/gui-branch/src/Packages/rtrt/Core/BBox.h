
#ifndef BBOX_H
#define BBOX_H 1

#include "Point.h"

namespace rtrt {

class BBox {
protected:
    Point cmin;
    Point cmax;
    bool have_some;
public:
    inline BBox() {
	have_some=false;
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
};

} // end namespace rtrt

#endif


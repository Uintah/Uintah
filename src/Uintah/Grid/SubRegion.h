
#ifndef UINTAH_HOMEBREW_SubRegion_H
#define UINTAH_HOMEBREW_SubRegion_H

#include "Array3Index.h"
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

class SubRegion {
public:
    SubRegion(const SCICore::Geometry::Point& lower,
	      const SCICore::Geometry::Point& upper,
	      int sx, int sy, int sz,
	      int ex, int ey, int ez);
    ~SubRegion();
    SubRegion(const SubRegion&);
    SubRegion& operator=(const SubRegion&);

    inline bool contains(const SCICore::Geometry::Vector& p) const {
	return p.x() >= lower.x() && p.y() >= lower.y() && p.z() >= lower.z()
	    && p.x() < upper.x() && p.y() < upper.y() && p.z() < upper.z();
    }
    inline bool contains(const Array3Index& idx) const {
	return idx.i() >= sx && idx.j() >= sy && idx.k() >= sz
	    && idx.i() <= ex && idx.j() <= ey && idx.k() <= ez;
    }
private:
    SCICore::Geometry::Point lower;
    SCICore::Geometry::Point upper;
    int sx, sy, sz;
    int ex, ey, ez;
};

#endif

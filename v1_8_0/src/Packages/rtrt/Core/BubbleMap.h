#ifndef BUBBLEMAP_H
#define BUBBLEMAP_H

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

class BubbleMap 
{
public:
    
    BubbleMap() {}

    Vector Perturbation(const Point &p) const
    {
	Point c(floor(p.x())+.5, floor(p.y())+.5, floor(p.z())+.5);
	Vector v = p-c;

	return v;
    }
    

    
};
} // end namespace rtrt

#endif


#include "SphereGeometryObject.h"
#include <SCICore/Geometry/Vector.h>

using namespace Uintah::Components;
using SCICore::Geometry::Vector;


SphereGeometryObject::SphereGeometryObject()
{
}

SphereGeometryObject::SphereGeometryObject(const double r, const  Point o):
  d_origin(o),d_radius(r)
{
}

SphereGeometryObject::~SphereGeometryObject()
{
}

void SphereGeometryObject::add(SphereGeometryObject* go)
{
}

bool SphereGeometryObject::inside(const Point& p) const
{
  Vector diff = p - d_origin;

  if (diff.length() > d_radius)
    return false;
  else 
    return true;
  
}

Box SphereGeometryObject::getBoundingBox() const
{
    Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
	   d_origin.z()-d_radius);

    Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
	   d_origin.z()+d_radius);

    return Box(lo,hi);

}


// $Log$
// Revision 1.1  2000/04/19 21:31:08  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//


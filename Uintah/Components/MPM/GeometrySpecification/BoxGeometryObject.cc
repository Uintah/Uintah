#include "BoxGeometryObject.h"
#include <SCICore/Math/MinMax.h>
#include "GeometryObjectFactory.h"
#include <SCICore/Geometry/Point.h>

using SCICore::Math::Min;
using SCICore::Math::Max;
using SCICore::Geometry::Point;

using namespace Uintah::Components;


BoxGeometryObject::BoxGeometryObject() {}

BoxGeometryObject::BoxGeometryObject(Point lo,Point up) :
  d_box(lo,up)
{
}

BoxGeometryObject::~BoxGeometryObject()
{
}

bool BoxGeometryObject::inside(const Point& p) const
{
  //Check p with the lower coordinates

  if (p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()) )
    return true;
  else
    return false;
	       
}

Box BoxGeometryObject::getBoundingBox() const
{
  return d_box;
}

GeometryObject* BoxGeometryObject::readParameters(ProblemSpecP &ps)
{
  Point min,max;
  ps->require("min",min);
  ps->require("max",max);
  
  // Not getting the resolution yet!

  return (new BoxGeometryObject(min,max));

}

// $Log$
// Revision 1.2  2000/04/20 15:09:25  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:07  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//

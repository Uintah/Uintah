#include "BoxGeometryObject.h"
#include "GeometryObjectFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Geometry/Point.h>
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
#include <string>

using namespace Uintah::Components;

BoxGeometryObject::BoxGeometryObject(ProblemSpecP& ps)
{
  Point min, max;
  ps->require("min",min);
  ps->require("max",max);  
  d_box=Box(min,max);
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

// $Log$
// Revision 1.4  2000/04/22 18:19:10  jas
// Filled in comments.
//
// Revision 1.3  2000/04/20 18:56:20  sparker
// Updates to MPM
//
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

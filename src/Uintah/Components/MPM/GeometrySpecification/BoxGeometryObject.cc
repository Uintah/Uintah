#include "BoxGeometryObject.h"
#include <SCICore/Math/MinMax.h>

using SCICore::Math::Min;
using SCICore::Math::Max;

using namespace Uintah::Components;


BoxGeometryObject::BoxGeometryObject() {}

BoxGeometryObject::BoxGeometryObject(Point lo,Point up) :
  d_box(lo,up)
{
}

BoxGeometryObject::~BoxGeometryObject()
{
}

void BoxGeometryObject::add(const BoxGeometryObject* go)
{
  // Need to fill in
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
// Revision 1.1  2000/04/19 21:31:07  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//

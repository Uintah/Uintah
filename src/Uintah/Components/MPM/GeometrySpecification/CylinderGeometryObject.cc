#include "CylinderGeometryObject.h"

using namespace Uintah::Components;


CylinderGeometryObject::CylinderGeometryObject() {}

CylinderGeometryObject::CylinderGeometryObject(CylinderGeometryObject::AXIS a,
					       Point o,
					       double l, double r) :
  d_axis(a),d_origin(o),d_length(l),d_radius(r)
{
}


CylinderGeometryObject::~CylinderGeometryObject()
{
}

void CylinderGeometryObject::add(const CylinderGeometryObject* go)
{

}

bool CylinderGeometryObject::inside(const Point &p) const
{

  // Do the x axis

  // Do the y axis 

  // Do the z axis

}

Box CylinderGeometryObject::getBoundingBox() const
{


}


// $Log$
// Revision 1.1  2000/04/19 21:31:07  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//

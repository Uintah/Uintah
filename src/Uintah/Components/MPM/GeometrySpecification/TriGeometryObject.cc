#include "TriGeometryObject.h"

using namespace Uintah::Components;

TriGeometryObject::TriGeometryObject()
{
}

TriGeometryObject::~TriGeometryObject()
{
}

void TriGeometryObject::add(TriGeometryObject *go)
{
}

bool TriGeometryObject::inside(const Point &p) const
{

}

Box TriGeometryObject::getBoundingBox() const
{

}



// $Log$
// Revision 1.1  2000/04/19 21:31:09  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//

#include "IntersectionGeometryObject.h"
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

using SCICore::Geometry::Point;
using SCICore::Math::Max;
using SCICore::Math::Min;

using namespace Uintah::Components;


IntersectionGeometryObject::IntersectionGeometryObject() 
{
}

IntersectionGeometryObject::IntersectionGeometryObject(const IntersectionGeometryObject& copy)
{
  // Need some help
}

IntersectionGeometryObject::~IntersectionGeometryObject()
{
  for (int i = 0; i < child.size(); i++) {
    delete child[i];
  }
}

void IntersectionGeometryObject::add(const GeometryObject *go)
{

  // How do I figure out the type of geometry object to create, use RTTI?
  GeometryObject* new_go;

  // Implement a factory method to create the new type
  
}


bool IntersectionGeometryObject::inside(const Point &p) const 
{
  for (int i = 0; i < child.size(); i++) {
    if (!child[i]->inside(p))
      return false;
  }
  return true;
}

Box IntersectionGeometryObject::getBoundingBox() const
{

  Point lo,hi;

  // Initialize the lo and hi points to the first element

  lo = child[0]->getBoundingBox().lower();
  hi = child[0]->getBoundingBox().upper();

  for (int i = 0; i < child.size(); i++) {
    Box box = child[i]->getBoundingBox();
    lo = Min(lo,box.lower());
    hi = Max(hi,box.upper());
  }

  return Box(lo,hi);
}


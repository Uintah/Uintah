#include "DifferenceGeometryObject.h"
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

using SCICore::Geometry::Point;
using SCICore::Math::Min;
using SCICore::Math::Max;

using namespace Uintah::Components;


DifferenceGeometryObject::DifferenceGeometryObject() 
{
}

DifferenceGeometryObject::DifferenceGeometryObject(const DifferenceGeometryObject& copy)
{
  // Need some help

}

DifferenceGeometryObject::~DifferenceGeometryObject()
{
  
  delete left;
  delete right;
 
}

void DifferenceGeometryObject::add(const GeometryObject *go)
{

  // How do I figure out the type of geometry object to create, use RTTI?
  GeometryObject* new_go;

  // Implement a factory method to create the new type
  
}


bool DifferenceGeometryObject::inside(const Point &p) const 
{
  return (left->inside(p) && !right->inside(p));
}

Box DifferenceGeometryObject::getBoundingBox() const
{

   // Initialize the lo and hi points to the left element

  Point left_lo = left->getBoundingBox().lower();
  Point left_hi = left->getBoundingBox().upper();
  Point right_lo = right->getBoundingBox().lower();
  Point right_hi = right->getBoundingBox().upper();
   
  Point lo = Min(left_lo,right_lo);
  Point hi = Max(left_hi,right_hi);
 
  return Box(lo,hi);
}


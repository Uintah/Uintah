#include "DifferenceGeometryObject.h"
#include <SCICore/Geometry/Point.h>

using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;

using namespace Uintah::Components;


DifferenceGeometryObject::DifferenceGeometryObject() 
{
}

DifferenceGeometryObject::~DifferenceGeometryObject()
{
  
  delete left;
  delete right;
 
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


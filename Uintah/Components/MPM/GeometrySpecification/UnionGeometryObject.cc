#include "UnionGeometryObject.h"
#include <SCICore/Geometry/Point.h>
#include "GeometryObjectFactory.h"

using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;

using namespace Uintah::Components;

UnionGeometryObject::UnionGeometryObject(ProblemSpecP &ps) 
{
  // Need to loop through all the geometry objects
  GeometryObjectFactory::create(ps,child);
  
}

UnionGeometryObject::~UnionGeometryObject()
{
  for (int i = 0; i < child.size(); i++) {
    delete child[i];
  }
}

bool UnionGeometryObject::inside(const Point &p) const 
{
  for (int i = 0; i < child.size(); i++) {
    if (child[i]->inside(p))
      return true;
  }
  return false;
}

Box UnionGeometryObject::getBoundingBox() const
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



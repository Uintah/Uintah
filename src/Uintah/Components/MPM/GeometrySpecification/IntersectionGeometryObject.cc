#include "IntersectionGeometryObject.h"
#include <SCICore/Geometry/Point.h>
#include "GeometryObjectFactory.h"

using SCICore::Geometry::Point;
using SCICore::Geometry::Max;
using SCICore::Geometry::Min;

using namespace Uintah::Components;


IntersectionGeometryObject::IntersectionGeometryObject(ProblemSpecP &ps) 
{
  GeometryObjectFactory::create(ps,child);

}

IntersectionGeometryObject::~IntersectionGeometryObject()
{
  for (int i = 0; i < child.size(); i++) {
    delete child[i];
  }
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


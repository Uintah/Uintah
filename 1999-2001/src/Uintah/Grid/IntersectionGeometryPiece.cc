#include "IntersectionGeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include "GeometryPieceFactory.h"

using SCICore::Geometry::Point;
using SCICore::Geometry::Max;
using SCICore::Geometry::Min;

using namespace Uintah;


IntersectionGeometryPiece::IntersectionGeometryPiece(ProblemSpecP &ps) 
{
  GeometryPieceFactory::create(ps,child);

}

IntersectionGeometryPiece::~IntersectionGeometryPiece()
{
  for (int i = 0; i < (int)child.size(); i++) {
    delete child[i];
  }
}

bool IntersectionGeometryPiece::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child.size(); i++) {
    if (!child[i]->inside(p))
      return false;
  }
  return true;
}

Box IntersectionGeometryPiece::getBoundingBox() const
{

  Point lo,hi;

  // Initialize the lo and hi points to the first element

  lo = child[0]->getBoundingBox().lower();
  hi = child[0]->getBoundingBox().upper();

  for (int i = 0; i < (int)child.size(); i++) {
    Box box = child[i]->getBoundingBox();
    lo = Min(lo,box.lower());
    hi = Max(hi,box.upper());
  }

  return Box(lo,hi);
}


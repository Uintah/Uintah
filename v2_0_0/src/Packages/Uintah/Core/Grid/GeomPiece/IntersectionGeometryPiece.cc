#include <Packages/Uintah/Core/Grid/GeomPiece/IntersectionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Core/Geometry/Point.h>

using namespace SCIRun;
using namespace Uintah;

IntersectionGeometryPiece::IntersectionGeometryPiece(ProblemSpecP &ps) 
{
  setName("intersection");
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


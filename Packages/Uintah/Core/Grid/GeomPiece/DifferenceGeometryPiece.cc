#include <Packages/Uintah/Core/Grid/GeomPiece/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>

#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <vector>

using namespace SCIRun;
using namespace Uintah;

DifferenceGeometryPiece::DifferenceGeometryPiece(ProblemSpecP &ps) 
{
  std::vector<GeometryPiece *> objs;

  GeometryPieceFactory::create(ps,objs);

  left = objs[0];
  right = objs[1];

}

DifferenceGeometryPiece::DifferenceGeometryPiece(GeometryPiece* p1,
						 GeometryPiece* p2)
  : left(p1), right(p2)
{
}

DifferenceGeometryPiece::~DifferenceGeometryPiece()
{
  
  delete left;
  delete right;
 
}

bool DifferenceGeometryPiece::inside(const Point &p) const 
{
  return (left->inside(p) && !right->inside(p));
}

Box DifferenceGeometryPiece::getBoundingBox() const
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

#include "DifferenceGeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include "GeometryPieceFactory.h"
#include <vector>

using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;

using namespace Uintah::MPM;
using namespace Uintah;

DifferenceGeometryPiece::DifferenceGeometryPiece(ProblemSpecP &ps) 
{
  std::vector<GeometryPiece *> objs(2);

  GeometryPieceFactory::create(ps,objs);

  left = objs[0];
  right = objs[1];

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





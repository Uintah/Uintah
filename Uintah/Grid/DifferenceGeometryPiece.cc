#include "DifferenceGeometryPiece.h"
#include <SCICore/Geometry/Point.h>
#include "GeometryPieceFactory.h"
#include <vector>
#include <iostream>

using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;

using namespace Uintah;
using std::cerr;
using std::endl;


DifferenceGeometryPiece::DifferenceGeometryPiece(ProblemSpecP &ps) 
{
  std::vector<GeometryPiece *> objs;

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
  std::cerr << "works here" << std::endl;

   // Initialize the lo and hi points to the left element

  Point left_lo = left->getBoundingBox().lower();
cerr << "left_lo = " << left_lo << endl;
  Point left_hi = left->getBoundingBox().upper();
  Point right_lo = right->getBoundingBox().lower();
  Point right_hi = right->getBoundingBox().upper();
   
  Point lo = Min(left_lo,right_lo);
  Point hi = Max(left_hi,right_hi);

cerr << "Point lo = " << lo << " hi = " << hi << endl;
 
  return Box(lo,hi);
}





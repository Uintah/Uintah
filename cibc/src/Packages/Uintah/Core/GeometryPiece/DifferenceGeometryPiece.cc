#include <Packages/Uintah/Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <vector>

using namespace SCIRun;
using namespace Uintah;

const string DifferenceGeometryPiece::TYPE_NAME = "difference";

DifferenceGeometryPiece::DifferenceGeometryPiece(ProblemSpecP &ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  std::vector<GeometryPieceP> objs;

  GeometryPieceFactory::create(ps,objs);

  left_  = objs[0];
  right_ = objs[1];

}

DifferenceGeometryPiece::DifferenceGeometryPiece(GeometryPieceP p1,
                                                 GeometryPieceP p2)
  : left_(p1), right_(p2)
{
  name_ = "Unnamed " + TYPE_NAME + " from pieces";
}

DifferenceGeometryPiece::~DifferenceGeometryPiece()
{
}

DifferenceGeometryPiece::DifferenceGeometryPiece(const DifferenceGeometryPiece& rhs)
{
  name_ = "Unnamed " + TYPE_NAME + " from CpyCnstr";

  left_  = rhs.left_->clone();
  right_ = rhs.right_->clone();
}

DifferenceGeometryPiece&
DifferenceGeometryPiece::operator=(const DifferenceGeometryPiece& rhs)
{
  if (this == &rhs)
    return *this;

  left_  = rhs.left_->clone();
  right_ = rhs.right_->clone();

  return *this;
}

void
DifferenceGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  left_->outputProblemSpec( ps );
  right_->outputProblemSpec( ps );
}

GeometryPieceP
DifferenceGeometryPiece::clone() const
{
  return scinew DifferenceGeometryPiece(*this);
}

bool
DifferenceGeometryPiece::inside(const Point &p) const 
{
  return (left_->inside(p) && !right_->inside(p));
}

Box
DifferenceGeometryPiece::getBoundingBox() const
{
   // Initialize the lo and hi points to the left element
  Point left_lo = left_->getBoundingBox().lower();
  Point left_hi = left_->getBoundingBox().upper();
  Point right_lo = right_->getBoundingBox().lower();
  Point right_hi = right_->getBoundingBox().upper();
   
  Point lo = Min(left_lo,right_lo);
  Point hi = Max(left_hi,right_hi);

  return Box(lo,hi);
}

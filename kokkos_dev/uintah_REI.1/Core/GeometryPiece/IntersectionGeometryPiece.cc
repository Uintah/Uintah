#include <Packages/Uintah/Core/GeometryPiece/IntersectionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

const string IntersectionGeometryPiece::TYPE_NAME = "intersection";

IntersectionGeometryPiece::IntersectionGeometryPiece(ProblemSpecP &ps) 
{
  name_ = "Unnamed Intersection";
  GeometryPieceFactory::create(ps,child_);

}

IntersectionGeometryPiece::IntersectionGeometryPiece(const IntersectionGeometryPiece& rhs)
{
  for( vector<GeometryPieceP>::const_iterator it = rhs.child_.begin();
       it != rhs.child_.end(); ++it )
    child_.push_back((*it)->clone());
}


IntersectionGeometryPiece::~IntersectionGeometryPiece()
{
}

IntersectionGeometryPiece&
IntersectionGeometryPiece::operator=(const IntersectionGeometryPiece& rhs)
{
  if (this == &rhs)
    return *this;

  child_.clear();

  // Copy in the new values
  for( vector<GeometryPieceP>::const_iterator it = rhs.child_.begin();
       it != rhs.child_.end(); ++it ) {
    child_.push_back((*it)->clone());
  }

  return *this;
}

void
IntersectionGeometryPiece::outputHelper( ProblemSpecP & ps) const
{
  for (vector<GeometryPieceP>::const_iterator it = child_.begin(); it != child_.end(); ++it) {
    (*it)->outputProblemSpec( ps );
  }
}

GeometryPieceP
IntersectionGeometryPiece::clone() const
{
  return scinew IntersectionGeometryPiece(*this);
}

bool
IntersectionGeometryPiece::inside(const Point &p) const 
{
  for( unsigned int i = 0; i < child_.size(); i++ ) {
    if (!child_[i]->inside(p))
      return false;
  }
  return true;
}

Box
IntersectionGeometryPiece::getBoundingBox() const
{
  Point lo,hi;

  // Initialize the lo and hi points to the first element

  lo = child_[0]->getBoundingBox().lower();
  hi = child_[0]->getBoundingBox().upper();

  for (unsigned int i = 0; i < child_.size(); i++) {
    Box box = child_[i]->getBoundingBox();
    lo = Min(lo,box.lower());
    hi = Max(hi,box.upper());
  }

  return Box(lo,hi);
}


#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

const string UnionGeometryPiece::TYPE_NAME = "union";

UnionGeometryPiece::UnionGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  // Need to loop through all the geometry pieces
  GeometryPieceFactory::create(ps,child_);
  
}

UnionGeometryPiece::UnionGeometryPiece(const vector<GeometryPieceP>& child) :
  child_(child)
{
  name_ = "Unnamed " + TYPE_NAME + " from vector";
}

UnionGeometryPiece&
UnionGeometryPiece::operator=(const UnionGeometryPiece& rhs){
  if (this == &rhs)
    return *this;

  child_.clear();

  // Copy in the new values
  for (vector<GeometryPieceP>::const_iterator it = rhs.child_.begin();
       it != rhs.child_.end(); ++it)
    child_.push_back((*it)->clone());

  return *this;
}

void
UnionGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  // If this is a named object, then only output the children the first time.
  for( vector<GeometryPieceP>::const_iterator it = child_.begin(); it != child_.end(); ++it ) {
    (*it)->outputProblemSpec( ps );
  }
}


GeometryPieceP
UnionGeometryPiece::clone() const
{
  return scinew UnionGeometryPiece(*this);
}

bool
UnionGeometryPiece::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child_.size(); i++) {
    if (child_[i]->inside(p)) {
      return true;
    }
  }
  return false;
}

Box UnionGeometryPiece::getBoundingBox() const
{

  Point lo,hi;

  // Initialize the lo and hi points to the first element

  lo = child_[0]->getBoundingBox().lower();
  hi = child_[0]->getBoundingBox().upper();

  for( unsigned int i = 0; i < child_.size(); i++ ) {
    Box box = child_[i]->getBoundingBox();
    lo = Min(lo,box.lower());
    hi = Max(hi,box.upper());
  }

  return Box(lo,hi);
}

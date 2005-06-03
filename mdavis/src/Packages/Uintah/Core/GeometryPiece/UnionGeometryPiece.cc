#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

UnionGeometryPiece::UnionGeometryPiece(ProblemSpecP &ps) 
{
  setName("union");
  // Need to loop through all the geometry pieces
  GeometryPieceFactory::create(ps,child);
  
}

UnionGeometryPiece::UnionGeometryPiece(const vector<GeometryPiece*>& child)
   : child(child)
{
}

UnionGeometryPiece::UnionGeometryPiece(const UnionGeometryPiece& rhs)
{
  for (vector<GeometryPiece*>::const_iterator it = rhs.child.begin();
       it != rhs.child.end(); ++it)
    child.push_back((*it)->clone());
}

UnionGeometryPiece& UnionGeometryPiece::operator=(const UnionGeometryPiece& rhs){
  if (this == &rhs)
    return *this;

  // Delete the lhs
  for (vector<GeometryPiece*>::const_iterator it = child.begin();
       it != child.end(); ++it)
    delete *it;
  child.clear();

  for (vector<GeometryPiece*>::const_iterator it = rhs.child.begin();
       it != rhs.child.end(); ++it)
    child.push_back((*it)->clone());

  return *this;

}

UnionGeometryPiece* UnionGeometryPiece::clone()
{
  return scinew UnionGeometryPiece(*this);
}


UnionGeometryPiece::~UnionGeometryPiece()
{
  for (int i = 0; i < (int)child.size(); i++) {
    delete child[i];
  }
}

bool UnionGeometryPiece::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child.size(); i++) {
    if (child[i]->inside(p))
      return true;
  }
  return false;
}

Box UnionGeometryPiece::getBoundingBox() const
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

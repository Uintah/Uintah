#include "UnionGeometryPiece.h"
#include <Core/Geometry/Point.h>
#include "GeometryPieceFactory.h"

using namespace SCIRun;
using namespace Uintah;
using namespace std;

UnionGeometryPiece::UnionGeometryPiece(ProblemSpecP &ps) 
{
  // Need to loop through all the geometry pieces
  GeometryPieceFactory::create(ps,child);
  
}

UnionGeometryPiece::UnionGeometryPiece(const vector<GeometryPiece*>& child)
   : child(child)
{
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

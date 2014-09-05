#include <Packages/Uintah/Core/GeometryPiece/IntersectionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Malloc/Allocator.h>
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

IntersectionGeometryPiece& IntersectionGeometryPiece::operator=(const IntersectionGeometryPiece& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  for (std::vector<GeometryPiece*>::const_iterator it = child.begin();
       it != child.end(); ++it)
    delete *it;
  child.clear();

  for (std::vector<GeometryPiece*>::const_iterator it = rhs.child.begin();
       it != rhs.child.end(); ++it)
    child.push_back((*it)->clone());

  return *this;

}


IntersectionGeometryPiece::IntersectionGeometryPiece(const IntersectionGeometryPiece& rhs)
{
  for (std::vector<GeometryPiece*>::const_iterator it = rhs.child.begin();
       it != rhs.child.end(); ++it)
    child.push_back((*it)->clone());

  
}

IntersectionGeometryPiece* IntersectionGeometryPiece::clone()
{
  return scinew IntersectionGeometryPiece(*this);
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


#include <Packages/Uintah/Core/Grid/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Point.h>

#include <string>

using namespace Uintah;
using namespace SCIRun;

BoxGeometryPiece::BoxGeometryPiece(ProblemSpecP& ps)
{
  Point min, max;
  ps->require("min",min);
  ps->require("max",max);  
  d_box=Box(min,max);
}

BoxGeometryPiece::~BoxGeometryPiece()
{
}

bool BoxGeometryPiece::inside(const Point& p) const
{
  //Check p with the lower coordinates

  if (p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()) )
    return true;
  else
    return false;
	       
}

Box BoxGeometryPiece::getBoundingBox() const
{
  return d_box;
}

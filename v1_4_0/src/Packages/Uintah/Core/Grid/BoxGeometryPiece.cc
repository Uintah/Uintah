#include <Packages/Uintah/Core/Grid/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Core/Geometry/Point.h>

#include <string>

using namespace Uintah;
using namespace SCIRun;

BoxGeometryPiece::BoxGeometryPiece(ProblemSpecP& ps)
{
  Point min, max;
  ps->require("min",min);
  ps->require("max",max); 
  
  double near_zero = 1e-100;
  double xdiff =  max.x() - min.x();
  double ydiff =  max.y() - min.y();
  double zdiff =  max.z() - min.z();
  
  if ( xdiff < near_zero   ||
       ydiff < near_zero   ||
       zdiff < near_zero ) {
    throw ProblemSetupException("Input File Error: box max <= min coordinates");
  }

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

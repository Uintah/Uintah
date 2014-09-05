#include <Packages/Uintah/Core/Grid/SphereGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace SCIRun;

SphereGeometryPiece::SphereGeometryPiece(ProblemSpecP& ps)
{

  Point orig;
  double rad;

  ps->require("origin",orig);
  ps->require("radius",rad);

  d_origin = orig;
  d_radius = rad;
}

SphereGeometryPiece::~SphereGeometryPiece()
{
}

bool SphereGeometryPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;

  if (diff.length() > d_radius)
    return false;
  else 
    return true;
  
}

Box SphereGeometryPiece::getBoundingBox() const
{
    Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
	   d_origin.z()-d_radius);

    Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
	   d_origin.z()+d_radius);

    return Box(lo,hi);

}



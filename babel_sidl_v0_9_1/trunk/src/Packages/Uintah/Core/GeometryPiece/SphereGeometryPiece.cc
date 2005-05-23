#include <Packages/Uintah/Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace SCIRun;


SphereGeometryPiece::SphereGeometryPiece(ProblemSpecP& ps)
{
  setName("sphere");
  Point orig = Point(0.,0.,0.);
  double rad = 0.;

  if(!ps->get("center", orig)) // Alternate specification
    ps->require("origin",orig);
  ps->require("radius",rad);
  
  if ( rad <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius must be > 0.0"));
  
  d_origin = orig;
  d_radius = rad;
}

SphereGeometryPiece::SphereGeometryPiece(const Point& origin,
		                         double radius)
{
  if ( radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius must be > 0.0"));
  
  d_origin = origin;
  d_radius = radius;
}

SphereGeometryPiece::~SphereGeometryPiece()
{
}

SphereGeometryPiece* SphereGeometryPiece::clone()
{
  return scinew SphereGeometryPiece(*this);
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



#include "BoxGeometryPiece.h"
#include "GeometryPieceFactory.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Geometry/Point.h>
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
#include <string>

using namespace Uintah::MPM;
using namespace Uintah;

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

// $Log$
// Revision 1.1  2000/06/09 18:38:20  jas
// Moved geometry piece stuff to Grid/ from MPM/GeometryPiece/.
//
// Revision 1.4  2000/04/26 06:48:22  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/04/24 21:04:28  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.4  2000/04/22 18:19:10  jas
// Filled in comments.
//
// Revision 1.3  2000/04/20 18:56:20  sparker
// Updates to MPM
//
// Revision 1.2  2000/04/20 15:09:25  jas
// Added factory methods for GeometryObjects.
//
// Revision 1.1  2000/04/19 21:31:07  jas
// Revamping of the way objects are defined.  The different geometry object
// subtypes only do a few simple things such as testing whether a point
// falls inside the object and also gets the bounding box for the object.
// The constructive solid geometry objects:union,difference, and intersection
// have the same simple operations.
//

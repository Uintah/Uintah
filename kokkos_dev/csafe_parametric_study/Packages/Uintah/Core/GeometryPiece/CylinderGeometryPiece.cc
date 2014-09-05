#include <Packages/Uintah/Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

const string CylinderGeometryPiece::TYPE_NAME = "cylinder";

CylinderGeometryPiece::CylinderGeometryPiece() 
{
  name_ = "Unnamed " + TYPE_NAME + " from BasicCtor";

  Point top, bottom;
  d_bottom = bottom;
  d_top = top;
  d_radius = 0.0;
}

CylinderGeometryPiece::CylinderGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  Point top,bottom;
  double rad;
  
  ps->require("bottom",bottom);
  ps->require("top",top);
  ps->require("radius",rad);
  
  double near_zero = 1e-100;
  Vector axis = top - bottom;
  
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("Input File Error: Cylinder axes has zero length", __FILE__, __LINE__));
  }
  if ( rad <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Cylinder radius must be > 0.0", __FILE__, __LINE__));
  }
  d_bottom = bottom;
  d_top = top;
  d_radius = rad;
}

CylinderGeometryPiece::CylinderGeometryPiece(const Point& top,
                                             const Point& bottom,
                                             double radius)
{
  name_ = "Unnamed " + TYPE_NAME + " from top/bottom/radius";

  double near_zero = 1e-100;
  Vector axis = top - bottom;
  
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("Input File Error: Cylinder axes has zero length", __FILE__, __LINE__));
  }
  if ( radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Cylinder radius must be > 0.0", __FILE__, __LINE__));
  }
  d_bottom = bottom;
  d_top = top;
  d_radius = radius;
}

CylinderGeometryPiece::~CylinderGeometryPiece()
{
}

void
CylinderGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("bottom",d_bottom);
  ps->appendElement("top",d_top);
  ps->appendElement("radius",d_radius);
}

GeometryPieceP
CylinderGeometryPiece::clone() const
{
  return scinew CylinderGeometryPiece(*this);
}

bool
CylinderGeometryPiece::inside(const Point &p) const
{
  Vector axis = d_top-d_bottom;  
  double height2 = axis.length2();

  Vector tobot = p-d_bottom;

  // pt is the "test" point
  double h = Dot(tobot, axis);
  if(h < 0.0 || h > height2)
    return false; // Above or below the cylinder

  double area = Cross(axis, tobot).length2();
  double d = area/height2;
  if( d > d_radius*d_radius)
    return false;
  return true;
}

Box
CylinderGeometryPiece::getBoundingBox() const
{
  Point lo(min(d_bottom.x(),d_top.x()) - d_radius,
           min(d_bottom.y(),d_top.y()) - d_radius,
           min(d_bottom.z(),d_top.z()) - d_radius);

  Point hi(max(d_bottom.x(),d_top.x()) + d_radius,
           max(d_bottom.y(),d_top.y()) + d_radius,
           max(d_bottom.z(),d_top.z()) + d_radius);

  return Box(lo,hi);
}

//////////
// Calculate the unit normal vector to axis from point
Vector 
CylinderGeometryPiece::radialDirection(const Point& pt) const
{
  Vector axis = d_top-d_bottom;  
  double height2 = axis.length();
  Vector pbot = pt-d_bottom;
  double tt = Dot(pbot,axis)/height2;
  Vector projOnAxis = d_bottom*(1.0-tt) + d_top*tt;;
  Vector normal = pt.asVector() - projOnAxis;
  return (normal/normal.length());
}


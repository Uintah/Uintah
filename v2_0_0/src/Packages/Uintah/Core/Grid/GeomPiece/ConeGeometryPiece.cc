#include <Packages/Uintah/Core/Grid/GeomPiece/ConeGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Vector.h>

#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ConeGeometryPiece::ConeGeometryPiece(ProblemSpecP& ps) 
{
  setName("cone");
  Point top, bottom;
  double topRad = 0.0;
  double botRad = 0.0;
  
  ps->require("bottom",bottom);
  ps->require("top",top);
  ps->get("bottom_radius",botRad);
  ps->get("top_radius",topRad);
  
  if (botRad == 0.0 && topRad == 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone volume == 0.0"));
  }
  double near_zero = 1e-100;
  Vector axis = top - bottom;
  
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone height == 0.0"));
  }
  if (botRad < 0.0 || topRad < 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone radius < 0.0"));
  }
  d_bottom = bottom;
  d_top = top;
  d_radius = botRad;
  d_topRad = topRad;
}

ConeGeometryPiece::ConeGeometryPiece(const Point& top,
		                     const Point& bottom,
				     double topRad,
                                     double botRad)
{
  if (botRad == 0.0 && topRad == 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone volume == 0.0"));
  }
  double near_zero = 1e-100;
  Vector axis = top - bottom;
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone height == 0.0"));
  }
  if (botRad < 0.0 || topRad < 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone radius < 0.0"));
  }
  d_bottom = bottom;
  d_top = top;
  d_radius = botRad;
  d_topRad = topRad;
}

ConeGeometryPiece::~ConeGeometryPiece()
{
}

bool 
ConeGeometryPiece::inside(const Point &pt) const
{
  // Find the position vector of top wrt bottom
  Vector axis = d_top-d_bottom;  
  double height2 = axis.length2();

  // Find the position vector of point wrt bottom
  Vector pbot = pt-d_bottom;

  // Project point on axis and find parametric location
  double tt = Dot(axis, pbot)/height2;

  // Above or below cone
  if (tt < 0.0 || tt > 1.0) return false;

  // Find the radius of the cross section of the cone
  // at this point
  double rad = d_radius*(1.0-tt) + d_topRad*tt;

  // Find the length of the vector from point to axis
  Vector projOnAxis = d_bottom*(1.0-tt) + d_top*tt;
  Vector normal = pt.asVector() - projOnAxis;
  double dist = normal.length();

  //cout << "Bottom = " << d_bottom << " Top = " << d_top << " Point = " << pt << endl;
  //cout << "tt = " << tt 
  //     << " Cur. Rad = " << rad << " Rad. Dist = " << dist << endl;

  // If dist < rad the point is inside
  if (dist > rad) return false;
  return true;
}

Box 
ConeGeometryPiece::getBoundingBox() const
{
  
  double rad = (d_radius > d_topRad) ? d_radius : d_topRad;
  Point lo(d_bottom.x() - rad, d_bottom.y() - rad,
	   d_bottom.z() - rad);

  Point hi(d_top.x() + rad, d_top.y() + rad,
	   d_top.z() + rad);

  return Box(lo,hi);
}

//////////
// Calculate the lateral surface area of the cone
double
ConeGeometryPiece::surfaceArea() const
{
  double rdiff = d_topRad - d_radius;
  double rsum = d_topRad + d_radius;
  double h = height();
  double s = sqrt(rdiff*rdiff + h*h);
  return (M_PI*s*rsum);
}

//////////
// Calculate the volume of the cone
double 
ConeGeometryPiece::volume() const
{
  double r2 = d_topRad*d_topRad;
  double rrp = d_topRad*d_radius;
  double rp2 = d_radius*d_radius;
  double h = height();
  return ((1.0/3.0)*M_PI*h*(r2+rrp+rp2));
}

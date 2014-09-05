#include <Packages/Uintah/Core/Grid/GeomPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Core/Geometry/Point.h>

#include <string>

#ifndef DMIN
#define DMIN(a,b) (a < b ? a : b)
#endif

using namespace Uintah;
using namespace SCIRun;

BoxGeometryPiece::BoxGeometryPiece(ProblemSpecP& ps)
{
  setName("box");
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
    SCI_THROW(ProblemSetupException("Input File Error: box max <= min coordinates"));
  }

  d_box=Box(min,max);
}

BoxGeometryPiece::BoxGeometryPiece(const Point& p1, const Point& p2)
  : d_box(Min(p1, p2), Max(p1, p2))
{
  if(d_box.degenerate())
    SCI_THROW(ProblemSetupException("degenerate box"));
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

double 
BoxGeometryPiece::volume() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  return (dx*dy*dz);
}

double 
BoxGeometryPiece::smallestSide() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  return DMIN(DMIN(dx,dy),dz);
}

unsigned int 
BoxGeometryPiece::thicknessDirection() const
{
  double dx = (d_box.upper()).x() - (d_box.lower()).x();
  double dy = (d_box.upper()).y() - (d_box.lower()).y();
  double dz = (d_box.upper()).z() - (d_box.lower()).z();
  double min = DMIN(DMIN(dx,dy),dz);
  if (dx == min) return 1;
  else if (dy == min) return 2;
  return 3;
}

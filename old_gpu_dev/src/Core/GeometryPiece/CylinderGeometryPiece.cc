/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;


const string CylinderGeometryPiece::TYPE_NAME = "cylinder";

CylinderGeometryPiece::CylinderGeometryPiece() 
{
  name_ = "Unnamed " + TYPE_NAME + " from BasicCtor";

  Point top, bottom;
  d_bottom = bottom;
  d_top = top;
  d_radius = 0.0;
  d_cylinder_end=false;
  d_axisymmetric_end=false;
  d_axisymmetric_side=false;
}

CylinderGeometryPiece::CylinderGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";
  Point top,bottom;
  double rad;
  
  ps->require("bottom",bottom);
  ps->require("top",top);
  ps->require("radius",rad);
  ps->getWithDefault("cylinder_end",     d_cylinder_end,      false);
  ps->getWithDefault("axisymmetric_end", d_axisymmetric_end,  false);
  ps->getWithDefault("axisymmetric_side",d_axisymmetric_side, false);
  
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
  d_cylinder_end=false;
  d_axisymmetric_end=false;
  d_axisymmetric_side=false;
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
  if(d_cylinder_end || d_axisymmetric_end || d_axisymmetric_side){
    ps->appendElement("cylinder_end",d_cylinder_end);
    ps->appendElement("axisymmetric_end", d_axisymmetric_end);
    ps->appendElement("axisymmetric_side",d_axisymmetric_side);
  }
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
  
  Point minBB = Min(d_bottom, d_top);
  Point maxBB = Max(d_bottom, d_top);
  
  double x_sqrd = pow( ( d_bottom.x() - d_top.x() ), 2);
  double y_sqrd = pow( ( d_bottom.y() - d_top.y() ), 2);
  double z_sqrd = pow( ( d_bottom.z() - d_top.z() ), 2);  
  double all_sqrd = x_sqrd + y_sqrd + z_sqrd;
  
  double kx = sqrt( (y_sqrd + z_sqrd)/all_sqrd );
  double ky = sqrt( (x_sqrd + z_sqrd)/all_sqrd );
  double kz = sqrt( (x_sqrd + y_sqrd)/all_sqrd );
  
  Vector tmp(kx*d_radius, ky*d_radius, kz*d_radius);
  minBB -= tmp;
  maxBB += tmp;

  return Box(minBB,maxBB);
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
  Vector projOnAxis = tt*axis/height2;
  Vector normal = pbot - projOnAxis;
  return (normal/normal.length());
}


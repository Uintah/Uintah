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


#include <Core/GeometryPiece/TorusGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

const string TorusGeometryPiece::TYPE_NAME = "torus";

TorusGeometryPiece::TorusGeometryPiece() 
{
  name_ = "Unnamed " + TYPE_NAME + " from BasicCtor";

  d_center = Point(0.,0.,0.);
  d_major_radius = 0.0;
  d_minor_radius = 0.0;
  d_axis = "x";
  d_theta = 0.0;
}

TorusGeometryPiece::TorusGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

  ps->require("center",d_center);
  ps->require("major_radius",d_major_radius);
  ps->require("minor_radius",d_minor_radius);
  ps->require("axis",d_axis);
  ps->getWithDefault("theta",d_theta,0.0);

  if ( d_minor_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus minor_radius must be > 0.0", __FILE__, __LINE__));
  }
  if ( d_major_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus major_radius must be > 0.0", __FILE__, __LINE__));
  }
  if(d_axis != "x" && d_axis != "y" && d_axis != "z"){
    SCI_THROW(ProblemSetupException("Input File Error: Torus axis must be 'x', 'y', or 'z'", __FILE__, __LINE__));
  }
}

TorusGeometryPiece::TorusGeometryPiece(const Point& center,
                                       const double major,
                                       const double minor,
                                       const string axis,
                                       const double theta)
{
  name_ = "Unnamed " + TYPE_NAME + " from center/major/minor";

  d_center = center;
  d_major_radius = major;
  d_minor_radius = minor;
  d_axis = axis;
  d_theta = theta;

  if ( d_minor_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus minor_radius must be > 0.0", __FILE__, __LINE__));
  }
  if ( d_major_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus major_radius must be > 0.0", __FILE__, __LINE__));
  }
  if(d_axis != "x" && d_axis != "y" && d_axis != "z"){
    SCI_THROW(ProblemSetupException("Input File Error: Torus axis must be 'x', 'y', or 'z'", __FILE__, __LINE__));
  }
}

TorusGeometryPiece::~TorusGeometryPiece()
{
}

void
TorusGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("center",d_center);
  ps->appendElement("major_radius",d_major_radius);
  ps->appendElement("minor_radius",d_minor_radius);
  ps->appendElement("axis",d_axis);
  ps->appendElement("rotation_angle",d_theta);
}

GeometryPieceP
TorusGeometryPiece::clone() const
{
  return scinew TorusGeometryPiece(*this);
}

bool
TorusGeometryPiece::inside(const Point &p) const
{
  double x = p.x() - d_center.x();
  double y = p.y() - d_center.y();
  double z = p.z() - d_center.z();
  if(d_axis=="z"){
    // rotate about the y-axis, i.e., keep y unchanged
    double xprime = x*cos(-d_theta) - z*sin(-d_theta);
    double zprime = x*sin(-d_theta) + z*cos(-d_theta);
    x=xprime; z=zprime;
    if((d_major_radius - sqrt(x*x + y*y))*
       (d_major_radius - sqrt(x*x + y*y)) + z*z <
        d_minor_radius*d_minor_radius){
      return true;
    }else{
      return false;
    }
  } // axis = z

  else if(d_axis=="y"){
    // rotate about the x-axis, i.e., keep x unchanged
    double yprime = y*cos(-d_theta) - z*sin(-d_theta);
    double zprime = y*sin(-d_theta) + z*cos(-d_theta);

    y=yprime; z=zprime;
    if((d_major_radius - sqrt(x*x + z*z))*
       (d_major_radius - sqrt(x*x + z*z)) + y*y <
        d_minor_radius*d_minor_radius){
      return true;
    }else{
      return false;
    }
  } // axis = y

  else if(d_axis=="x"){
    // rotate about the z-axis, i.e., keep z unchanged
    double xprime = x*cos(-d_theta) - y*sin(-d_theta);
    double yprime = x*sin(-d_theta) + y*cos(-d_theta);
    x=xprime; y=yprime;
    if((d_major_radius - sqrt(y*y + z*z))*
       (d_major_radius - sqrt(y*y + z*z)) + x*x <
        d_minor_radius*d_minor_radius){
      return true;
    }else{
      return false;
    }
  } // axis = x
  else{
    SCI_THROW(ProblemSetupException("Input File Error: Torus axis must be 'x', 'y', or 'z'", __FILE__, __LINE__));
  }

}

Box
TorusGeometryPiece::getBoundingBox() const
{
  // This is an overly generous bounding box.
  double R = d_major_radius+d_minor_radius;
  Point minBB = Point(d_center.x()-R,d_center.y()-R,d_center.z()-R);
  Point maxBB = Point(d_center.x()+R,d_center.y()+R,d_center.z()+R);
  
  return Box(minBB,maxBB);
}

//////////
// Calculate the unit normal vector to axis from point
Vector 
TorusGeometryPiece::radialDirection(const Point& pt) const
{
  // The following is WRONG, it is just a placeholder until necessity
  // dictates that I implement the correct version, which is a bit painful
  return pt - d_center;
}

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
}

TorusGeometryPiece::TorusGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

  ps->require("center",d_center);
  ps->require("major_radius",d_major_radius);
  ps->require("minor_radius",d_minor_radius);

  if ( d_minor_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus minor_radius must be > 0.0", __FILE__, __LINE__));
  }
  if ( d_major_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus major_radius must be > 0.0", __FILE__, __LINE__));
  }
}

TorusGeometryPiece::TorusGeometryPiece(const Point& center,
                                       const double major,
                                       const double minor)
{
  name_ = "Unnamed " + TYPE_NAME + " from center/major/minor";

  d_center = center;
  d_major_radius = major;
  d_minor_radius = minor;

  if ( d_minor_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus minor_radius must be > 0.0", __FILE__, __LINE__));
  }
  if ( d_major_radius <= 0.0) {
    SCI_THROW(ProblemSetupException("Input File Error: Torus major_radius must be > 0.0", __FILE__, __LINE__));
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
  if((d_major_radius - sqrt(x*x + y*y))*
     (d_major_radius - sqrt(x*x + y*y)) + z*z <
      d_minor_radius*d_minor_radius){
    return true;
  }else{
/*
  if((d_major_radius - sqrt(p.x()*p.x() + p.y()*p.y()))*
     (d_major_radius - sqrt(p.x()*p.x() + p.y()*p.y())) + p.z()*p.z() <
      d_minor_radius*d_minor_radius){
    return true;
  }else{
*/
    return false;
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

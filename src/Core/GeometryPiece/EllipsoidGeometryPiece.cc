/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <Core/GeometryPiece/EllipsoidGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Matrix3.h>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

const string EllipsoidGeometryPiece::TYPE_NAME = "ellipsoid";

EllipsoidGeometryPiece::EllipsoidGeometryPiece(ProblemSpecP& ps)
{
  name_ = "Unnamed " + TYPE_NAME + " from PS";

  Vector zero = Vector(0.,0.,0.);

  ps->require("origin",d_origin);
  // Get Vector axes
  ps->getWithDefault("v1",d_v1,     zero);
  ps->getWithDefault("v2",d_v2,     zero);
  ps->getWithDefault("v3",d_v3,     zero);
  
  // Get orthagonal axes
  ps->getWithDefault("rx",d_radiusX, 0.0);
  ps->getWithDefault("ry",d_radiusY, 0.0);
  ps->getWithDefault("rz",d_radiusZ, 0.0);
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::EllipsoidGeometryPiece(const Point& origin,
                                               double radx, double rady, double radz )
{
  d_origin = origin;
  d_radiusX = radx;
  d_radiusY = rady;
  d_radiusZ = radz;
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::EllipsoidGeometryPiece(const Point& origin,
                                               Vector one, Vector two, Vector three )
{
  d_origin = origin;
  d_v1 = one;
  d_v2 = two;
  d_v3 = three;
  
  // Run helper function to determine if inputs are correct
  initializeEllipsoidData();
}

EllipsoidGeometryPiece::~EllipsoidGeometryPiece()
{
}

void EllipsoidGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("origin",d_origin);
  if(xyzAligned) { // only need to output radii
    ps->appendElement("rx",    d_radiusX);
    ps->appendElement("ry",    d_radiusX);
    ps->appendElement("rz",    d_radiusX);
  } else {
    ps->appendElement("v1",    d_v1);
    ps->appendElement("v2",    d_v2);
    ps->appendElement("v3",    d_v3);
  }
}

GeometryPieceP EllipsoidGeometryPiece::clone() const
{
  return scinew EllipsoidGeometryPiece(*this);
}

bool EllipsoidGeometryPiece::inside(const Point& p) const
{
  // Variable initialization
  Point *pRotated = new Point(p.x()-d_origin.x(),p.y()-d_origin.y(),p.z()-d_origin.z());
  
  // create rotate
  if(!xyzAligned)
  {
    // Rotate point
    // Note, angles are negated so that it's opposite of what ellipse is rotated
    pRotated = new Point(cos(-thetaz)*pRotated->x() - sin(-thetaz)*pRotated->y(), 
                         cos(-thetaz)*pRotated->y() + sin(-thetaz)*pRotated->x(), 
                         pRotated->z());
    pRotated = new Point(cos(-thetay)*pRotated->x() + sin(-thetay)*pRotated->z(), 
                         pRotated->y(), 
                         cos(-thetay)*pRotated->z() - sin(-thetay)*pRotated->x());
    pRotated = new Point(pRotated->x(),
                         cos(-thetax)*pRotated->y() + sin(-thetax)*pRotated->z(),
                         cos(-thetax)*pRotated->z() - sin(-thetax)*pRotated->y());
  }
  
  // Check if in unit distance from sphere center of sphere after scaling
  if (sqrt(pRotated->x()*pRotated->x()/(d_radiusX*d_radiusX) +
           pRotated->y()*pRotated->y()/(d_radiusY*d_radiusY) +
           pRotated->z()*pRotated->z()/(d_radiusZ*d_radiusZ)) <= 1.0) {
    return true;                           
  } else {
    return false;
  }
}

Box EllipsoidGeometryPiece::getBoundingBox() const
{
  double highX = 0.0;
  double highY = 0.0;
  double highZ = 0.0;

  // Use vectors to find highest xyz
  // X
  highX = d_v1.x();
  if(abs(d_v2.x()) > highX)
    highX = d_v2.x();
  if(abs(d_v3.x()) > highX)
    highX = d_v3.x();
  // Y
  highY = d_v1.y();
  if(abs(d_v2.y()) > highY)
    highY = d_v2.y();
  if(abs(d_v3.y()) > highY)
    highY = d_v3.y();
  // X
  highZ = d_v1.z();
  if(abs(d_v2.z()) > highZ)
    highZ = d_v2.z();
  if(abs(d_v3.z()) > highZ)
    highZ = d_v3.z();
  
    Point low( d_origin.x()-abs(highX),d_origin.y()-abs(highY),
           d_origin.z()-abs(highZ) );

    Point high( d_origin.x()+abs(highX),d_origin.y()+abs(highY),
           d_origin.z()+abs(highZ) );

    return Box(low,high);
}

void EllipsoidGeometryPiece::initializeEllipsoidData()
{
  // determine whether input is from vector or double
  if(d_v1.length() > 0.0 &&
     d_v2.length() > 0.0 &&
     d_v3.length() > 0.0){
    // Check for orthagonality
    if(Dot(d_v1,d_v2) > 1e-12 ||
       Dot(d_v2,d_v3) > 1e-12 ||
       Dot(d_v3,d_v1) > 1e-12 )
    {
      throw new ProblemSetupException("Input vectors are not orthagonal to within 1e-12 or each other", __FILE__, __LINE__, false);
    }
    
    // compute radius of each vector when aligned to grid
    d_radiusX = d_v1.length();
    d_radiusY = d_v2.length();
    d_radiusZ = d_v3.length();
    
    
    // Initialize variables for rotation
    double xnew = 0.0, ynew = 0.0, znew = 0.0;
    thetaz = 0.0, thetay = 0.0, thetax = 0.0;
    
    Vector unitX = *(new Vector(1.0,0.0,0.0));
    Vector unitY = *(new Vector(0.0,1.0,0.0));
    Vector unitZ = *(new Vector(0.0,0.0,1.0));
    Vector temporary = *(new Vector(d_v1));
    
    // Compute degree to which it is rotated
    // Find rotation about Z
    Vector projection = temporary - unitZ*(Dot(unitZ,temporary));
    if(projection[0] > 0.0)
      thetaz = atan(projection[1]/projection[0]);
    else 
      thetaz = 0.0;
    
    xnew = cos(thetaz)*(temporary[0]) - sin(thetaz)*(temporary[1]);
    ynew = cos(thetaz)*(temporary[1]) + sin(thetaz)*(temporary[0]);
    temporary = *(new Vector(xnew,ynew,(temporary[2])));
    
    // Find rotation about Y
    projection = (temporary) - (unitY)*(Dot((unitY),(temporary)));
    if(projection[0] > 0.0 )
      thetay = atan(projection[2]/projection[0]);
    else
      thetay = 0.0;
    
    znew = cos(thetay)*(temporary[2]) - sin(thetay)*(temporary[0]);
    xnew = cos(thetay)*(temporary[0]) + sin(thetay)*(temporary[2]);
    temporary = *(new Vector(xnew,temporary[1],znew));
    
    // Find rotation about X
    projection = temporary - unitX*(Dot(unitX,temporary));
    if(projection[1] > 0.0)
      thetax = atan(projection[2]/projection[1]);
    else
      thetax = 0.0;
    
    znew = cos(thetax)*(temporary[2]) - sin(thetax)*(temporary[1]);
    ynew = cos(thetax)*(temporary[1]) + sin(thetax)*(temporary[2]);
    temporary = *(new Vector(temporary[0],ynew,znew));
    
    xyzAligned = false;
  } else if(d_radiusX > 0.0 &&
            d_radiusY > 0.0 &&
            d_radiusZ > 0.0){
    // create vector representation along the cartesian axes
    d_v1 = *(new Vector(d_radiusX, 0.0, 0.0));
    d_v2 = *(new Vector(0.0, d_radiusY, 0.0));
    d_v3 = *(new Vector(0.0, 0.0, d_radiusZ));
    
    // set flag such that rotation doesnt need to occur in inside()
    xyzAligned = true;
      
  } else {
      throw new ProblemSetupException("Input must have axis lengths greater than zero.", __FILE__, __LINE__, false );
  }
}

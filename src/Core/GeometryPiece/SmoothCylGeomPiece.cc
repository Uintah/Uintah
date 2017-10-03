/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/GeometryPiece/SmoothCylGeomPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Math/Matrix3.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace std;
using namespace Uintah;

//______________________________________________________________________
//    Note:  This is not intended to be used with the union, difference intersection
//           operators.
//______________________________________________________________________


const string SmoothCylGeomPiece::TYPE_NAME = "smoothcyl";

//////////
// Constructor : Initialize stuff
SmoothCylGeomPiece::SmoothCylGeomPiece(ProblemSpecP& ps)
{
  ps->require("discretization_scheme", d_discretization);
  if (d_discretization != "pie_slices" && d_discretization != "constant_particle_volumes") 
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Invalid discretization scheme (valid options are pie_slices and constant_particle_volumes)", __FILE__, __LINE__));

  ps->require("bottom", d_bottom);
  ps->require("top", d_top);
  if ((d_top-d_bottom).length2() <= 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Check data in input", __FILE__, __LINE__));

  ps->require("outer_radius", d_outer_radius);
  if (d_outer_radius <= 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Outer Radius < 0", __FILE__, __LINE__));

  ps->require("num_radial", d_numRadial);
  if (d_numRadial < 1)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Radial Divs < 1", __FILE__, __LINE__));

  ps->require("num_axial", d_numAxial);
  if (d_numAxial < 1)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Axial Divs < 1", __FILE__, __LINE__));

  d_numAngular = 0;
  ps->get("num_angular", d_numAngular);
  if (d_discretization == "constant_particle_volumes") {
    cout << "**WARNING** num_angular ignored for constant_particle_volumes discretization" << endl;
  } else if (d_discretization == "pie_slices") {
    if (d_numAngular < 1)
      SCI_THROW(ProblemSetupException("SmoothCylGeom: Angular Divs < 1", __FILE__, __LINE__));
  }

  d_inner_radius = 0.0;
  ps->get("inner_radius", d_inner_radius);
  if (d_inner_radius > d_outer_radius)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Inner Radius > Outer Radius", __FILE__, __LINE__));
  if (d_inner_radius < 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Inner Radius < 0.0", __FILE__, __LINE__));

  d_capThick = 0.0;
  ps->get("endcap_thickness", d_capThick);
  if (d_capThick < 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Cap Thickness < 0.0", __FILE__, __LINE__));

  d_arcStart = 0.0;
  double arcStart = 0.0;
  ps->get("arc_start_angle", arcStart);
  if (arcStart > 0.0) d_arcStart = (M_PI/180.0)*arcStart;
  if (d_arcStart < 0.0 || d_arcStart > 2.0*M_PI)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Arc Start Angle < 0.0 || > 2*Pi", __FILE__, __LINE__ ));

  d_angle = 2.0*M_PI;
  double angle = -1.0;
  ps->get("arc_angle", angle);
  if (angle > 0.0) d_angle = (M_PI/180.0)*angle;
  if (d_angle < 0.0 || d_angle > 2.0*M_PI)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Angle < 0.0 || > 360", 
			   __FILE__, __LINE__ ));

  d_fileName = "none";
  ps->get("output_file", d_fileName);

}

//////////
// Destructor
SmoothCylGeomPiece::~SmoothCylGeomPiece()
{
}

void
SmoothCylGeomPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("bottom", d_bottom);
  ps->appendElement("top", d_top);
  ps->appendElement("outer_radius", d_outer_radius);
  ps->appendElement("inner_radius", d_inner_radius);
  ps->appendElement("num_radial", d_numRadial);
  ps->appendElement("num_axial", d_numAxial);
  ps->appendElement("num_angular", d_numAngular);
  ps->appendElement("endcap_thickness", d_capThick);
  ps->appendElement("arc_start_angle", d_arcStart);
  ps->appendElement("arc_angle", d_angle);
  ps->appendElement("output_file", d_fileName);
  ps->appendElement("discretization_scheme", d_discretization);
}

GeometryPieceP
SmoothCylGeomPiece::clone() const
{
  return scinew SmoothCylGeomPiece(*this);
}

/////////////////////////////////////////////////////////////////////////////
/*! Find if a point is inside the cylinder or end caps */
/////////////////////////////////////////////////////////////////////////////
bool 
SmoothCylGeomPiece::inside(const Point& p) const
{
  bool isInside = true;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double length = axis.length();
  Vector capAxis = axis*(d_capThick/length);

  // a) Check is the point is within the solid composite cylinder
  Vector bot = d_bottom.asVector() - capAxis;
  Vector top = d_top.asVector() + capAxis;
  Vector fullAxis = top - bot;
  double height  = length + 2*d_capThick;
  double height2 = height*height;
  Vector pToBot = p.asVector() - bot;
  double h = Dot(pToBot, fullAxis);
  if(h < 0.0 || h > height2) isInside = false;
  double area = Cross(fullAxis, pToBot).length2();
  double d = area/height2;
  if( d > d_outer_radius*d_outer_radius) isInside = false;

  // b) Find if the point is outside the inner cylinder
  if (isInside) {
    pToBot = p - d_bottom;
    area = Cross(axis, pToBot).length2();
    d = area/(length*length);
    if(!(d > d_inner_radius*d_inner_radius)) isInside = false;
  }
  return isInside;
}

/////////////////////////////////////////////////////////////////////////////
/*! Find the bounding box for the cylinder */
/////////////////////////////////////////////////////////////////////////////
Box 
SmoothCylGeomPiece::getBoundingBox() const
{
  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  Vector capAxis = axis*(d_capThick/axis.length());

  Vector bot = d_bottom.asVector() - capAxis;
  Vector top = d_top.asVector() + capAxis;
  Point lo(bot.x() - d_outer_radius, bot.y() - d_outer_radius,
           bot.z() - d_outer_radius);
  Point hi(top.x() + d_outer_radius, top.y() + d_outer_radius,
           top.z() + d_outer_radius);

  return Box(lo,hi);
}

//////////////////////////////////////////////////////////////////////////
/* Create particles */
//////////////////////////////////////////////////////////////////////////
unsigned int 
SmoothCylGeomPiece::createPoints()
{
  int totCount = 0;
  if (d_capThick > 0.0) {
    int count = createEndCapPoints();
    totCount += count;
  }

  int count = createCylPoints();
  totCount += count;

  // Write the output if requested
  if (d_fileName != "none") {
    writePoints(d_fileName, "pts");
    writePoints(d_fileName, "vol");
  }

  return totCount;
}

//////////////////////////////////////////////////////////////////////////
/*! Create the particles on a circle on the x-y plane and then
  rotate them to the correct position. First particle is located 
  at the center. */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createEndCapPoints()
{
  cout << "Creating particles for the End Caps" << endl;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double axislen = axis.length();
  axis /= axislen;

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, axis));

  // Rotation axis
  Vector a = Cross(n0, axis);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Initialize count of the number of particles
  int count = 0;

  // Calculate the radial and axial material point spacing
  double axisInc = axislen/(double) d_numAxial;
  int numCapAxial = int(d_capThick/axisInc)-1;
  double radInc = d_outer_radius/(double) d_numRadial;

  // Create particles for the bottom end cap
  double currZ = 0.5*axisInc;
  for (int kk = 0; kk < numCapAxial; ++kk) {
    Vector currCenter = d_bottom.asVector() - axis*currZ;

    // Put a point at the center
    //d_points.push_back(Point(currCenter));
    //double area = 0.25*M_PI*radInc*radInc;
    //d_volume.push_back(axisInc*area);
    //count++;
    
    for (int ii = 0; ii < d_numRadial; ++ii) {
      double prevRadius = ii*radInc;
      double currRadius = prevRadius + 0.5*radInc;
      double nextRadius = (ii+1)*radInc;
      //int numCircum = (int) (2.0*M_PI*currRadius/radInc);
      //double phiInc = 2.0*M_PI/(double) numCircum;
      int numCircum = (int) (d_angle*currRadius/radInc);
      double phiInc = d_angle/(double) numCircum;
      double area = 0.5*phiInc*(nextRadius*nextRadius-prevRadius*prevRadius);
      for (int jj = 0; jj < numCircum; ++jj) {
        double phi = d_arcStart + jj*phiInc; 
        double cosphi = cos(phi);
        double sinphi = sin(phi);

        // Create points on xy plane
        double x = currRadius*cosphi;
        double y = currRadius*sinphi;
        double z = 0;
     
        // Rotate points to correct orientation and
        // Translate to correct position
        Vector pp(x, y, z);
        pp = R*pp + currCenter;
        Point p(pp);

        d_points.push_back(p);
        d_volume.push_back(axisInc*area);
        //cout << "Point["<<count<<"]="<<p<<endl;
        count++;
      }
    }
    currZ -= axisInc;
  }
  
  // Create particles for the top end cap
  currZ = 0.5*axisInc;
  for (int kk = 0; kk < numCapAxial; ++kk) {
    Vector currCenter = d_top.asVector() + axis*currZ;

    // Put a point at the center
    //d_points.push_back(Point(currCenter));
    //double area = 0.25*M_PI*radInc*radInc;
    //d_volume.push_back(axisInc*area);
    //count++;
    
    for (int ii = 0; ii < d_numRadial; ++ii) {
      double prevRadius = ii*radInc;
      double currRadius = prevRadius + 0.5*radInc;
      double nextRadius = (ii+1)*radInc;
      //int numCircum = (int) (2.0*M_PI*currRadius/radInc);
      //double phiInc = 2.0*M_PI/(double) numCircum;
      int numCircum = (int) (d_angle*currRadius/radInc);
      double phiInc = d_angle/(double) numCircum;
      double area = 0.5*phiInc*(nextRadius*nextRadius-prevRadius*prevRadius);
      for (int jj = 0; jj < numCircum; ++jj) {
        double phi = d_arcStart + jj*phiInc; 
        double cosphi = cos(phi);
        double sinphi = sin(phi);

        // Create points on xy plane
        double x = currRadius*cosphi;
        double y = currRadius*sinphi;
        double z = 0;
     
        // Rotate points to correct orientation and
        // Translate to correct position
        Vector pp(x, y, z);
        pp = R*pp + currCenter;
        Point p(pp);

        d_points.push_back(p);
        d_volume.push_back(axisInc*area);
        //cout << "Point["<<count<<"]="<<p<<endl;
        count++;
      }
    }
    currZ += axisInc;
  }
  
  return count;
}

//////////////////////////////////////////////////////////////////////////
/*! Create the particles on a circle on the x-y plane and then
  rotate them to the correct position */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createCylPoints()
{
  //cout << "Creating particles for the Cylinder" << endl;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double length = axis.length();
  axis = axis/length;

  double cell_vol = d_DX.x()*d_DX.y()*d_DX.z();

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, axis));

  // Rotation axis
  Vector a = Cross(n0, axis);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Initialize count of the number of particles
  int count = 0;

  // Calculate the radial and axial material point spacing

  double thickness = d_outer_radius - d_inner_radius;
  double axisInc = length/(double) d_numAxial;
  double radInc = thickness/(double) d_numRadial;
  double innerRad = d_inner_radius;
  int numAngular = d_numAngular;

  d_points.clear();
  d_volume.clear();
  d_size.clear();
  // Create particles for the hollow cylinder
  double currZ = 0.5*axisInc;
  for (int kk = 0; kk < d_numAxial; ++kk) {
    Vector currCenter = d_bottom.asVector() + axis*currZ;
    for (int ii = 0; ii < d_numRadial; ++ii) {
      double prevRadius = innerRad + ii*radInc;
      double r = prevRadius + radInc*0.5;
      double nextRadius = innerRad + (ii+1)*radInc;
      double angularInc=0.0;
      double area;
   
      if (d_discretization == "constant_particle_volumes") {
	numAngular = (int) ((d_angle - d_arcStart)*r/radInc);
	numAngular = max(numAngular,1);
        angularInc = (d_angle - d_arcStart)/(double) numAngular;
        area = 0.5*angularInc*(nextRadius*nextRadius-prevRadius*prevRadius);
      } else {
        bool tooMany=true;
        double redNumAng=1.;
        while(tooMany){
          numAngular = d_numAngular*redNumAng;
  
          angularInc = (d_angle - d_arcStart)/(double) numAngular;
          area = 0.5*angularInc*(nextRadius*nextRadius-prevRadius*prevRadius);
          double pvol = area*axisInc;
          if(pvol/cell_vol < 5.e-2){
            redNumAng*=0.9;
            tooMany=true;
          } else {
            tooMany=false;
          }
        }
      }

      for (int jj = 0; jj < numAngular; ++jj) {
	double phi = d_arcStart + (jj+0.5)*angularInc; 
	double cosphi = cos(phi);
	double sinphi = sin(phi);

	// Create points on xy plane
	double x = r*cosphi;
	double y = r*sinphi;
	double z = 0;
	Matrix3 size((d_angle - d_arcStart)*y/(numAngular*d_DX.x()),  thickness*x/(d_numRadial*r*d_DX.y()), 0.0,
		     -(d_angle - d_arcStart)*x/(numAngular*d_DX.x()), thickness*y/(d_numRadial*r*d_DX.y()), 0.0,
		     0.0                                            , 0.0                                   , length/(d_numAxial*d_DX.z()));

	// Rotate points to correct orientation and
	// Translate to correct position
	Vector pp(x, y, z);
	pp = R*pp + currCenter;
	Point p(pp);

	// Rotate size matrix
	size = R*size*R.Transpose();

	d_points.push_back(p);
	d_volume.push_back(axisInc*area);
	d_size.push_back(size);
        // area vector contains three components, these are:
        // (area normal to r, area normal to circumference, area normal to axis)
//	d_area.push_back(Vector(axisInc*r*angularInc,radInc*axisInc,area));

        // area vector contains three components, these are:
        // (area normal to r, area normal to r, area normal to axis)
	d_area.push_back(Vector(axisInc*r*angularInc,
                                axisInc*r*angularInc, area));
	//cout << "Size["<<count<<"] = "<<d_size[count]<<endl;
	count++;
      }
    }
    currZ += axisInc;
  } 

  return count;
}

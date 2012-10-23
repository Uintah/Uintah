/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/GeometryPiece/ConeGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

#ifndef M_PI
#  define M_PI           3.14159265358979323846  /* pi */
#endif


using namespace Uintah;
using namespace SCIRun;

const string ConeGeometryPiece::TYPE_NAME = "cone";

ConeGeometryPiece::ConeGeometryPiece(ProblemSpecP& ps) 
{
  name_ = "Unnamed Cone";
  Point top, bottom;
  double topRad = 0.0;
  double botRad = 0.0;
  
  ps->require("bottom",bottom);
  ps->require("top",top);
  ps->get("bottom_radius",botRad);
  ps->get("top_radius",topRad);
  
  if (botRad == 0.0 && topRad == 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone volume == 0.0", __FILE__, __LINE__));
  }
  double near_zero = 1e-100;
  Vector axis = top - bottom;
  
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone height == 0.0", __FILE__, __LINE__));
  }
  if (botRad < 0.0 || topRad < 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone radius < 0.0", __FILE__, __LINE__));
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
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone volume == 0.0", __FILE__, __LINE__));
  }
  double near_zero = 1e-100;
  Vector axis = top - bottom;
  if ( axis.length()  < near_zero ) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone height == 0.0", __FILE__, __LINE__));
  }
  if (botRad < 0.0 || topRad < 0.0) {
    SCI_THROW(ProblemSetupException("** INPUT ERROR ** Cone radius < 0.0", __FILE__, __LINE__));
  }
  d_bottom = bottom;
  d_top = top;
  d_radius = botRad;
  d_topRad = topRad;
}

ConeGeometryPiece::~ConeGeometryPiece()
{
}

GeometryPieceP
ConeGeometryPiece::clone() const
{
  return scinew ConeGeometryPiece(*this);
}

void
ConeGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("bottom",d_bottom);
  ps->appendElement("top",d_top);
  ps->appendElement("bottom_radius",d_radius);
  ps->appendElement("top_radius",d_topRad);
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

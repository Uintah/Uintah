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

#include <Core/Grid/BoundaryConditions/EllipseBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <iostream>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

// export SCI_DEBUG="BC_dbg:+"
static DebugStream BC_dbg("BC_dbg",false);

EllipseBCData::EllipseBCData() : BCGeomBase()
{}

EllipseBCData::EllipseBCData(Point& p, double minorRadius,double majorRadius, const std::string face, double angleDegrees)
  : BCGeomBase(), d_origin(p), 
    d_minorRadius(minorRadius), 
    d_majorRadius(majorRadius),  
    d_angleDegrees(angleDegrees),
    d_face(face)
{}

EllipseBCData::~EllipseBCData()
{}


bool EllipseBCData::operator==(const BCGeomBase& rhs) const
{
  const EllipseBCData* p_rhs = 
    dynamic_cast<const EllipseBCData*>(&rhs);
  
  if (p_rhs == NULL)
    return false;
  else 
    return (this->d_minorRadius  == p_rhs->d_minorRadius  ) &&
           (this->d_majorRadius  == p_rhs->d_majorRadius  ) &&
           (this->d_origin       == p_rhs->d_origin       ) &&
           (this->d_face         == p_rhs->d_face         ) &&
           (this->d_angleDegrees == p_rhs->d_angleDegrees );
}

EllipseBCData* EllipseBCData::clone()
{
  return scinew EllipseBCData(*this);
}

void EllipseBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void EllipseBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void EllipseBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool EllipseBCData::inside(const Point &p) const 
{
  Point f1 (d_origin);
  Point f2 (d_origin);
  const double pi = 3.141592653589793;
  const double angleRad = d_angleDegrees*pi/180.0;
  double ellipseFormula =0.0;
  const double focalDistance = sqrt(d_majorRadius*d_majorRadius - d_minorRadius*d_minorRadius);

  if (d_face=="x-") {        
    f1.y(d_origin.y() + focalDistance*cos(angleRad));
    f1.z(d_origin.z() - focalDistance*sin(angleRad));    
    f2.y(d_origin.y() - focalDistance*cos(angleRad));
    f2.z(d_origin.z() + focalDistance*sin(angleRad));    
  } 
  
  else if (d_face=="x+") {
    f1.y(d_origin.y() - focalDistance*cos(angleRad));
    f1.z(d_origin.z() - focalDistance*sin(angleRad));    
    f2.y(d_origin.y() + focalDistance*cos(angleRad));
    f2.z(d_origin.z() + focalDistance*sin(angleRad));        
  } 
  
  else if (d_face=="y-") {
    f1.x(d_origin.x() - focalDistance*sin(angleRad));
    f1.z(d_origin.z() + focalDistance*cos(angleRad));    
    f2.x(d_origin.x() + focalDistance*sin(angleRad));
    f2.z(d_origin.z() - focalDistance*cos(angleRad));    
  } 
  
  else if (d_face=="y+") {
    f1.x(d_origin.x() + focalDistance*sin(angleRad));
    f1.z(d_origin.z() + focalDistance*cos(angleRad));    
    f2.x(d_origin.x() - focalDistance*sin(angleRad));
    f2.z(d_origin.z() - focalDistance*cos(angleRad));    
  }
  
  else if (d_face=="z-") {
    f1.x(d_origin.x() + focalDistance*cos(angleRad));
    f1.y(d_origin.y() - focalDistance*sin(angleRad));    
    f2.x(d_origin.x() - focalDistance*cos(angleRad));
    f2.y(d_origin.y() + focalDistance*sin(angleRad));    
  } 
  
  else if (d_face=="z+") {
    f1.x(d_origin.x() + focalDistance*cos(angleRad));
    f1.y(d_origin.y() + focalDistance*sin(angleRad));    
    f2.x(d_origin.x() - focalDistance*cos(angleRad));
    f2.y(d_origin.y() - focalDistance*sin(angleRad));        
  }
  
  Vector diff1 = p - f1;
  Vector diff2 = p - f2;
  ellipseFormula = diff1.length() + diff2.length() - 2.0*d_majorRadius;  
  return (ellipseFormula <= 0.0);
}

void EllipseBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}

void EllipseBCData::determineIteratorLimits(Patch::FaceType face, 
                                           const Patch* patch, 
                                           vector<Point>& test_pts)
{
#if 0
  cout << "Ellipse determineIteratorLimits()" << endl;
#endif

  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}


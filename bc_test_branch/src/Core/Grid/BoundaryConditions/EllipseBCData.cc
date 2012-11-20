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
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

EllipseBCData::EllipseBCData( const Point           & origin,
                                    double            minorRadius,
                                    double            majorRadius,
                                    double            angleDegrees,
                              const string          & name,
                              const Patch::FaceType & side ) :
  BCGeomBase( name, side ), d_origin( origin ), d_minorRadius( minorRadius ), d_majorRadius( majorRadius ), d_angleDegrees( angleDegrees )
{}

EllipseBCData::~EllipseBCData()
{}

bool
EllipseBCData::operator==(const BCGeomBase& rhs) const
{
  const EllipseBCData* p_rhs = 
    dynamic_cast<const EllipseBCData*>(&rhs);
  
  if (p_rhs == NULL)
    return false;
  else 
    return (this->d_minorRadius  == p_rhs->d_minorRadius  ) &&
           (this->d_majorRadius  == p_rhs->d_majorRadius  ) &&
           (this->d_origin       == p_rhs->d_origin       ) &&
           (this->d_angleDegrees == p_rhs->d_angleDegrees );
}

bool
EllipseBCData::inside( const Point & pt ) const 
{
  Point f1 (d_origin);
  Point f2 (d_origin);

  const double angleRad = d_angleDegrees*M_PI/180.0;
  double ellipseFormula =0.0;
  const double focalDistance = sqrt(d_majorRadius*d_majorRadius - d_minorRadius*d_minorRadius);

  if( d_faceSide == Patch::xminus ) {
    f1.y(d_origin.y() + focalDistance*cos(angleRad));
    f1.z(d_origin.z() - focalDistance*sin(angleRad));    
    f2.y(d_origin.y() - focalDistance*cos(angleRad));
    f2.z(d_origin.z() + focalDistance*sin(angleRad));    
  } 
  
  else if( d_faceSide == Patch::xplus ) {
    f1.y(d_origin.y() - focalDistance*cos(angleRad));
    f1.z(d_origin.z() - focalDistance*sin(angleRad));    
    f2.y(d_origin.y() + focalDistance*cos(angleRad));
    f2.z(d_origin.z() + focalDistance*sin(angleRad));        
  } 
  
  else if( d_faceSide == Patch::yminus ) {
    f1.x(d_origin.x() - focalDistance*sin(angleRad));
    f1.z(d_origin.z() + focalDistance*cos(angleRad));    
    f2.x(d_origin.x() + focalDistance*sin(angleRad));
    f2.z(d_origin.z() - focalDistance*cos(angleRad));    
  } 
  
  else if( d_faceSide == Patch::yplus ) {
    f1.x(d_origin.x() + focalDistance*sin(angleRad));
    f1.z(d_origin.z() + focalDistance*cos(angleRad));    
    f2.x(d_origin.x() - focalDistance*sin(angleRad));
    f2.z(d_origin.z() - focalDistance*cos(angleRad));    
  }
  
  else if( d_faceSide == Patch::zminus ) {
    f1.x(d_origin.x() + focalDistance*cos(angleRad));
    f1.y(d_origin.y() - focalDistance*sin(angleRad));    
    f2.x(d_origin.x() - focalDistance*cos(angleRad));
    f2.y(d_origin.y() + focalDistance*sin(angleRad));    
  } 
  
  else if( d_faceSide == Patch::zplus ) {
    f1.x(d_origin.x() + focalDistance*cos(angleRad));
    f1.y(d_origin.y() + focalDistance*sin(angleRad));    
    f2.x(d_origin.x() - focalDistance*cos(angleRad));
    f2.y(d_origin.y() - focalDistance*sin(angleRad));        
  }
  
  Vector diff1 = pt - f1;
  Vector diff2 = pt - f2;
  ellipseFormula = diff1.length() + diff2.length() - 2.0*d_majorRadius;  
  return (ellipseFormula <= 0.0);
}

void
EllipseBCData::print( int depth ) const
{
  string indentation( depth, ' ' );
  cout << indentation << "EllipseBCData\n";
  for( map<int,BCData*>::const_iterator itr = d_bcs.begin(); itr != d_bcs.end(); itr++ ) {
    itr->second->print( depth + 2 );
  }
}

void
EllipseBCData::determineIteratorLimits(       Patch::FaceType   face, 
                                        const Patch           * patch, 
                                              vector<Point>   & test_pts )
{
  BCGeomBase::determineIteratorLimits( face, patch, test_pts );
}


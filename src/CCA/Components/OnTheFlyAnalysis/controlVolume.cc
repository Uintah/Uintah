/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/controlVolume.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <sstream>
#include <cstdio>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//
controlVolume::controlVolume( const ProblemSpecP& boxps,
                              const GridP& grid )
{
  ProblemSpecP box_ps = boxps;

  if( box_ps == nullptr ){
    throw ProblemSetupException("ERROR: OnTheFlyAnalysis/Control Volume:  Couldn't find <controlVolumes> -> <box> tag", __FILE__, __LINE__);
  }

  map<string,string> attribute;
  box_ps->getAttributes(attribute);
  m_CV_name = attribute["label"];

  Point min;
  Point max;
  box_ps->require("min",min);
  box_ps->require("max",max);

  double near_zero = 10 * DBL_MIN;
  double xdiff =  std::fabs( max.x() - min.x() );
  double ydiff =  std::fabs( max.y() - min.y() );
  double zdiff =  std::fabs( max.z() - min.z() );

  if ( xdiff < near_zero   ||
       ydiff < near_zero   ||
       zdiff < near_zero ) {
    std::ostringstream warn;
    warn << "\nERROR: OnTheFlyAnalysis/Control Volume: box ("<< m_CV_name
         << ") The max coordinate cannot be <= the min coordinate (max " << max << " <= min " << min << ")." ;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  Box cannot exceed comutational boundaries
  BBox b;
  grid->getInteriorSpatialRange( b );

  const Point bmin = b.min();
  const Point bmax = b.max();

  if( min.x() < bmin.x() || min.y() < bmin.y() || min.x() < bmin.z() ||
      max.x() > bmax.x() || max.y() > bmax.y() || max.x() > bmax.z() ){
    proc0cout << "______________________________________________________________________\n"
              << " DataAnalysis:\n"
              << " WARNING:  The extents of controlVolume (" << m_CV_name << ") exceed the computational domain.\n"
              << "           Resizing the box so it doesn't exceed the domain.\n"
              << "______________________________________________________________________";
  }

  min = Max( b.min(), min );
  max = Min( b.max(), max );
  m_box = Box(min,max);
}

//______________________________________________________________________
//
controlVolume::~controlVolume(){};


//______________________________________________________________________
//
void controlVolume::initialize( const Level* level)
{  
  m_lowIndx = findCell( level, m_box.lower() );
  m_highIndx = findCell( level, m_box.upper() );
  
  proc0cout << " ControlVolume: " << m_CV_name << " " << m_lowIndx << " " << m_highIndx << "\n";

  // bulletproofing
  IntVector diff = m_highIndx - m_lowIndx;

  if ( diff.x() < 1 || diff.y() < 1 || diff.z() < 1 ){
    std::ostringstream warn;
    warn << "\nERROR: OnTheFlyAnalysis/Control Volume: box ("<< m_CV_name
           << ") There must be at least one computational cell difference between the max and min ("<< diff << ")";
   throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//  Cell iterator over all cells in the control volumne in this patch
CellIterator controlVolume::getCellIterator( const Patch* patch ) const {

  IntVector lo = Max(m_lowIndx,  patch->getCellLowIndex());
  IntVector hi = Min(m_highIndx, patch->getCellHighIndex());

  return CellIterator( lo, hi );
}

//______________________________________________________________________
//
CellIterator controlVolume::getFaceIterator(const controlVolume::FaceType& face,
                                            const FaceIteratorType& domain,
                                            const Patch* patch) const
{
  IntVector lowPt  = m_lowIndx;
  IntVector highPt = m_highIndx;

  // principle direction
  const int pDir = getFaceAxes(face)[0];

  bool plusface= (face == xplus || face == yplus || face == zplus);

  switch(domain)
  {
    case SFC_Cells:         // offset the plus faces by 1 cell

      if(plusface){
        lowPt[pDir]  =  highPt[pDir];
        highPt[pDir] += 1;
      }
      else{
        highPt[pDir] = lowPt[pDir] + 1;
      }
      break;

    case InteriorFaceCells:

      if(plusface){
        lowPt[pDir] = highPt[pDir] - 1;
      }
      else{
        highPt[pDir] = lowPt[pDir] + 1;
      }
      break;
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }

  IntVector pLo = patch->getExtraCellLowIndex();
  IntVector pHi = patch->getExtraCellHighIndex();
  
  IntVector lo = Max(lowPt,  pLo);
  IntVector hi = Min(highPt, pHi);

  IntVector diff = hi - lo;
  if ( diff.x() < 1 || diff.y() < 1 || diff.z() < 1 ){
    std::ostringstream warn;
    warn << "\nERROR: OnTheFlyAnalysis/Control Volume: getFaceIterator ("<< m_CV_name
         << ") There must be at least one computational cell in each direction ("<< diff << ")"
         << "\n p_lo " << pLo << " pHi: " << pHi
         << "\n lo:  " << lo  << " hi:  " << hi;
   throw InternalError(warn.str(), __FILE__, __LINE__);
  }

  return CellIterator(lo, hi);

}

//______________________________________________________________________
// Returns the normal to the patch face
Vector controlVolume::getFaceNormal( const controlVolume::FaceType & face ) const
{
  switch( face )
  {
    case xminus:
      return Vector(-1,0,0);
    case xplus:
      return Vector(1,0,0);
    case yminus:
      return Vector(0,-1,0);
    case yplus:
      return Vector(0,1,0);
    case zminus:
      return Vector(0,0,-1);
    case zplus:
      return Vector(0,0,1);
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
// Sets the vector faces equal to the list of faces that are on the boundary
void controlVolume::getBoundaryFaces( std::vector<FaceType>& faces,
                                      const Patch* patch) const
{
  faces.clear();

  IntVector p_lo =  patch->getExtraCellLowIndex();
  IntVector p_hi =  patch->getExtraCellHighIndex();

  bool doesIntersect_XY = (m_highIndx.x() > p_lo.x() &&
                           m_lowIndx.x()  < p_hi.x() &&
                           m_highIndx.y() > p_lo.y() &&
                           m_lowIndx.y()  < p_hi.y() );

  bool doesIntersect_XZ = (m_highIndx.x() > p_lo.x() &&
                           m_lowIndx.x()  < p_hi.x() &&
                           m_highIndx.z() > p_lo.z() &&
                           m_lowIndx.z()  < p_hi.z() );

  bool doesIntersect_YZ = (m_highIndx.y() > p_lo.y() &&
                           m_lowIndx.y()  < p_hi.y() &&
                           m_highIndx.z() > p_lo.z() &&
                           m_lowIndx.z()  < p_hi.z() );
       
  if( m_lowIndx.x()  >= p_lo.x()  && doesIntersect_YZ ) {
    faces.push_back( xminus );
  }
  if( m_highIndx.x()  < p_hi.x() && doesIntersect_YZ ) {
    faces.push_back( xplus );
  }
  if( m_lowIndx.y()  >= p_lo.y()  && doesIntersect_XZ ) {
    faces.push_back( yminus );
  }
  if( m_highIndx.y()  < p_hi.y() && doesIntersect_XZ ) {
    faces.push_back( yplus );
  }
  if( m_lowIndx.z()  >= p_lo.z()  && doesIntersect_XY ) {
    faces.push_back( zminus );
  }
  if( m_highIndx.z()  < p_hi.z() && doesIntersect_XY ) {
    faces.push_back( zplus );
  }
}

//______________________________________________________________________
//
IntVector controlVolume::findCell( const Level * level, 
                                   const Point & p)
{
  Vector  dx     = level->dCell();
  Point   anchor = level->getAnchor();
  Vector  v( (p - anchor) / dx);
  
  IntVector cell (roundNearest(v));  // This is slightly different from Level implementation
  return cell;
}

//______________________________________________________________________
//
std::string
controlVolume::getExtents_string() const
{
  ostringstream mesg;
  mesg.setf(ios::scientific,ios::floatfield);
  mesg.precision(4);
  mesg << "  controlVolume (" << m_CV_name
      << ") box lower: " << m_box.lower() << " upper: " << m_box.upper()
      <<", lowIndex: "  << m_lowIndx << ", highIndex: " << m_highIndx;
  mesg.setf(ios::scientific ,ios::floatfield);

  return mesg.str();
}

//______________________________________________________________________
// Returns the principal axis along a face and
// the orthognonal axes to that face (right hand rule).
IntVector controlVolume::getFaceAxes( const controlVolume::FaceType & face ) const
{
  switch(face)
  {
    case xminus:
    case xplus:
      return IntVector(0,1,2);
    case yminus:
    case yplus:
      return IntVector(1,2,0);
    case zminus:
    case zplus:
      return IntVector(2,0,1);
    default:
      throw InternalError("Invalid FaceType in controlVolume::getfaceAxes", __FILE__, __LINE__);
  };
}

//______________________________________________________________________
//
string
controlVolume::getFaceName(controlVolume::FaceType face) const
{
  switch(face) {
  case xminus:
    return "xminus";
  case xplus:
    return "xplus";
  case yminus:
    return "yminus";
  case yplus:
    return "yplus";
  case zminus:
    return "zminus";
  case zplus:
    return "zplus";
  default:
    SCI_THROW(InternalError("Illegal FaceType in controlVolume::getFaceName", __FILE__, __LINE__));
  }
}

 //______________________________________________________________________
 // Returns the cell area dx*dy.
double controlVolume::getCellArea( const controlVolume::FaceType face,
                                   const Patch* patch ) const
{
   double area = 0.0;
   Vector dx = patch->dCell();

   switch (face) {
     case xminus:
     case xplus:
       area  = dx.y() * dx.z();
       break;
     case yminus:
     case yplus:
       area  = dx.x() * dx.z();
       break;
     case zminus:
     case zplus:
       area  = dx.x() * dx.y();
       break;
     default:
       break;
   };
   return area;
 }

//______________________________________________________________________
//
void
controlVolume::print()
{
  DOUT( true, getExtents_string() );
}



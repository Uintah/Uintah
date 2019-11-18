/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Box.h>

#include <iostream>
#include <sstream>
#include <cstdio>

using namespace Uintah;
using namespace std;

controlVolume::controlVolume( const IntVector & lowIndex,
                              const IntVector & highIndex )
  : d_lowIndex(lowIndex)
  , d_highIndex(highIndex)
{
}
//______________________________________________________________________
//  Cell iterator over all cells in the control volumne in this patch
inline CellIterator controlVolume::getCellIterator( const Patch* patch ) const {

  IntVector lo = Max(d_lowIndex,  patch->getCellLowIndex());
  IntVector hi = Min(d_highIndex, patch->getCellHighIndex());
  
  return CellIterator( lo, hi ); 
}

//______________________________________________________________________
//
CellIterator controlVolume::getFaceIterator(const controlVolume::FaceType& face, 
                                            const FaceIteratorType& domain,
                                            const Patch* patch) const
{
  IntVector lowPt  = d_lowIndex; 
  IntVector highPt = d_highIndex;

  //compute the dimension
  int dim=face/2;

  //compute if we are a plus face
  bool plusface=face%2;

  switch(domain)
  {
    case MinusEdgeCells:

      //select the face
      if(plusface){
        lowPt[dim]  =  highPt[dim];   //restrict dimension to the face
        highPt[dim] += 1;             //extend dimension by extra cells
      }
      else{
        highPt[dim] = lowPt[dim];     //restrict dimension to face
        lowPt[dim]  -=1;              //extend dimension by extra cells
      }
      break;

    case InteriorFaceCells:
      
      //select the face
      if(plusface){
        lowPt[dim] = highPt[dim];     //restrict index to face
        lowPt[dim]--;                 //contract dimension by 1
      }
      else{
        highPt[dim] = lowPt[dim];     //restrict index to face
        highPt[dim]++;                //contract dimension by 1
      }
      break;
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }

  IntVector lo = Max(lowPt,  patch->getCellLowIndex());
  IntVector hi = Min(highPt, patch->getCellHighIndex());
  
  return CellIterator(lo, hi);

}

//______________________________________________________________________
// Returns the normal to the patch face
IntVector controlVolume::getFaceDirection( const controlVolume::FaceType & face ) const
{
  switch( face )
  {
    case xminus:
      return IntVector(-1,0,0);
    case xplus:
      return IntVector(1,0,0);
    case yminus:
      return IntVector(0,-1,0);
    case yplus:
      return IntVector(0,1,0);
    case zminus:
      return IntVector(0,0,-1);
    case zplus:
      return IntVector(0,0,1);
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
// Sets the vector faces equal to the list of faces that are on the boundary
inline void controlVolume::getBoundaryFaces(std::vector<FaceType>& faces,
                                            const Patch* patch) const
{
  faces.clear();

  IntVector p_lo =  patch->getCellLowIndex();
  IntVector p_hi =  patch->getCellHighIndex();

  bool doesIntersect_XY = (d_highIndex.x() > p_lo.x() &&
                           d_lowIndex.x()  < p_hi.x() &&
                           d_highIndex.y() > p_lo.y() &&
                           d_lowIndex.y()  < p_hi.y() ); 

  bool doesIntersect_XZ = (d_highIndex.x() > p_lo.x() &&
                           d_lowIndex.x()  < p_hi.x() &&
                           d_highIndex.z() > p_lo.z() &&
                           d_lowIndex.z()  < p_hi.z() ); 

  bool doesIntersect_YZ = (d_highIndex.y() > p_lo.y() &&
                           d_lowIndex.y()  < p_hi.y() &&
                           d_highIndex.z() > p_lo.z() &&
                           d_lowIndex.z()  < p_hi.z() );

  if( d_lowIndex.x()  > p_lo.x()  && doesIntersect_YZ ) {
    faces.push_back( xminus );
  }
  if( d_highIndex.x()  < p_hi.x() && doesIntersect_YZ ) {
    faces.push_back( xplus );
  }     
  if( d_lowIndex.y()  > p_lo.y()  && doesIntersect_XZ ) {
    faces.push_back( yminus );
  }
  if( d_highIndex.y()  < p_hi.y() && doesIntersect_XZ ) {
    faces.push_back( yplus );
  }
  if( d_lowIndex.z()  > p_lo.z()  && doesIntersect_XY ) {
    faces.push_back( zminus );
  }
  if( d_highIndex.z()  < p_hi.z() && doesIntersect_XY ) {
    faces.push_back( zplus );
  }
}

//______________________________________________________________________
//  
std::string
controlVolume::toString(const Patch* patch) const
{

  Box box = patch->getLevel()->getBox(d_lowIndex, d_highIndex);
  
  char str[ 1024 ];
  sprintf( str, "[ [%2.2f, %2.2f, %2.2f] [%2.2f, %2.2f, %2.2f] ]",
           box.lower().x(), box.lower().y(), box.lower().z(),
           box.upper().x(), box.upper().y(), box.upper().z() );

  return string( str );
}

//______________________________________________________________________
// Returns the principal axis along a face and
// the orthognonal axes to that face (right hand rule).
IntVector controlVolume::getFaceAxes( const controlVolume::FaceType & face ) const
{
  switch(face)
  {
    case xminus: case xplus:
      return IntVector(0,1,2);
    case yminus: case yplus:
      return IntVector(1,2,0);
    case zminus: case zplus:
      return IntVector(2,0,1);
    default:
      throw InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
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
    SCI_THROW(InternalError("Illegal FaceType in controlVolume::faceName", __FILE__, __LINE__));
  }
}

 //______________________________________________________________________
 // Returns the cell area dx*dy.
double controlVolume::cellArea( const controlVolume::FaceType face,
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





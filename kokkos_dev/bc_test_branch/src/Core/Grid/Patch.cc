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

#include <Core/Grid/Patch.h>

#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/Box.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Primes.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>

#include <TauProfilerForSCIRun.h>

#include <iostream>
#include <sstream>
#include <cstdio>
#include <map>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

static AtomicCounter ids("Patch ID counter",0);
static Mutex ids_init("ID init");

Patch::Patch( const Level* level,
              const IntVector& lowIndex, const IntVector& highIndex,
              const IntVector& inLowIndex, const IntVector& inHighIndex, 
              unsigned int levelIndex,  int id ) :
  d_lowIndex(inLowIndex), d_highIndex(inHighIndex), 
  d_grid(0), d_id(id) , d_realPatch(0), d_level_index(-1),
  d_arrayBCS(0)
{
  
  if(d_id == -1){
    d_id = ids++;

  } else {
    if(d_id >= ids)
      ids.set(d_id+1);
  }
   
  // DON'T call setBCType here     
  d_patchState.xminus = None;
  d_patchState.yminus = None;
  d_patchState.zminus = None;
  d_patchState.xplus  = None;
  d_patchState.yplus  = None;
  d_patchState.zplus  = None;

  // Set the level index:
  d_patchState.levelIndex = levelIndex;
}

Patch::Patch(const Patch* realPatch, const IntVector& virtualOffset) :
  d_lowIndex(realPatch->getCellLowIndex()+virtualOffset),
  d_highIndex(realPatch->getCellHighIndex()+virtualOffset),
  d_grid(realPatch->d_grid),
  d_realPatch(realPatch), d_level_index(realPatch->d_level_index),
  d_arrayBCS(realPatch->d_arrayBCS)
{
  //if(!ids){
  // make the id be -1000 * realPatch id - some first come, first serve index
  d_id = -1000 * realPatch->d_id; // temporary
  int index = 1;
  int numVirtualPatches = 0;
  
  //set the level index
  d_patchState.levelIndex=realPatch->d_patchState.levelIndex;

  for (Level::const_patchIterator iter = getLevel()->allPatchesBegin(); iter != getLevel()->allPatchesEnd(); iter++) {
    if ((*iter)->d_realPatch == d_realPatch) {
      ++index;
      if (++numVirtualPatches >= 27)
        SCI_THROW(InternalError("A real patch shouldn't have more than 26 (3*3*3 - 1) virtual patches",
                                __FILE__, __LINE__));
    }
  }

  d_id -= index;
  
  //set boundary conditions
  d_patchState.xminus=realPatch->getBCType(xminus);
  d_patchState.yminus=realPatch->getBCType(yminus);
  d_patchState.zminus=realPatch->getBCType(zminus);
  d_patchState.xplus=realPatch->getBCType(xplus);
  d_patchState.yplus=realPatch->getBCType(yplus);
  d_patchState.zplus=realPatch->getBCType(zplus);
}

Patch::~Patch()
{
  if( d_arrayBCS ) {
    for( Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face=Patch::nextFace(face) ) {
      delete (*d_arrayBCS)[face];
    }
    delete d_arrayBCS;
  }
}

/**
* Returns the 8 nodes found around the point pos
*/
void
Patch::findCellNodes( const Point& pos, IntVector ni[8] ) const
{
  Point cellpos = getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
}

/**
 * Returns the 27 nodes found around the point pos
 */
void
Patch::findCellNodes27( const Point& pos, IntVector ni[27] ) const
{
  cerr << "findCellNodes27 appears to be incorrect.  You are using it at your own risk" << endl;
  Point cellpos = getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  int nnx,nny,nnz;

  if(cellpos.x()-(ix) <= .5){ nnx = -1; } else{ nnx = 2; }
  if(cellpos.y()-(iy) <= .5){ nny = -1; } else{ nny = 2; }
  if(cellpos.z()-(iz) <= .5){ nnz = -1; } else{ nnz = 2; }

  ni[0]  = IntVector(ix,    iy,      iz);
  ni[1]  = IntVector(ix+1,  iy,      iz);
  ni[2]  = IntVector(ix+nnx,iy,      iz);
  ni[3]  = IntVector(ix,    iy+1,    iz);
  ni[4]  = IntVector(ix+1,  iy+1,    iz);
  ni[5]  = IntVector(ix+nnx,iy+1,    iz);
  ni[6]  = IntVector(ix,    iy+nny,  iz);
  ni[7]  = IntVector(ix+1,  iy+nny,  iz);
  ni[8]  = IntVector(ix+nnx,iy+nny,  iz);
  ni[9]  = IntVector(ix,    iy,      iz+1);
  ni[10] = IntVector(ix+1,  iy,      iz+1);
  ni[11] = IntVector(ix+nnx,iy,      iz+1);
  ni[12] = IntVector(ix,    iy+1,    iz+1);
  ni[13] = IntVector(ix+1,  iy+1,    iz+1);
  ni[14] = IntVector(ix+nnx,iy+1,    iz+1);
  ni[15] = IntVector(ix,    iy+nny,  iz+1);
}


/**
 * Returns the position of the node idx in domain coordinates.
 */
Point
Patch::nodePosition(const IntVector& idx) const
{
  return getLevel()->getNodePosition(idx);
}

/**
 * Returns the position of the cell idx in domain coordinates.
 */
Point
Patch::cellPosition(const IntVector& idx) const
{
  return getLevel()->getCellPosition(idx);
}

void
Patch::findCellsFromNode( const IntVector & nodeIndex,
                                IntVector   cellIndex[8] ) 
{
   int ix = nodeIndex.x();
   int iy = nodeIndex.y();
   int iz = nodeIndex.z();

   cellIndex[0] = IntVector(ix, iy, iz);
   cellIndex[1] = IntVector(ix, iy, iz-1);
   cellIndex[2] = IntVector(ix, iy-1, iz);
   cellIndex[3] = IntVector(ix, iy-1, iz-1);
   cellIndex[4] = IntVector(ix-1, iy, iz);
   cellIndex[5] = IntVector(ix-1, iy, iz-1);
   cellIndex[6] = IntVector(ix-1, iy-1, iz);
   cellIndex[7] = IntVector(ix-1, iy-1, iz-1);
}

void
Patch::findNodesFromCell( const IntVector & cellIndex,
                                IntVector   nodeIndex[8] )
{
   int ix = cellIndex.x();
   int iy = cellIndex.y();
   int iz = cellIndex.z();

   nodeIndex[0] = IntVector(ix, iy, iz);
   nodeIndex[1] = IntVector(ix, iy, iz+1);
   nodeIndex[2] = IntVector(ix, iy+1, iz);
   nodeIndex[3] = IntVector(ix, iy+1, iz+1);
   nodeIndex[4] = IntVector(ix+1, iy, iz);
   nodeIndex[5] = IntVector(ix+1, iy, iz+1);
   nodeIndex[6] = IntVector(ix+1, iy+1, iz);
   nodeIndex[7] = IntVector(ix+1, iy+1, iz+1);
}

namespace Uintah {
  ostream&
  operator<<(ostream& out, const Patch & r)
  {
    out.setf(ios::scientific,ios::floatfield);
    out.precision(4);
    out << "(Patch " << r.getID() 
        << ", lowIndex=" << r.getExtraCellLowIndex() << ", highIndex=" 
        << r.getExtraCellHighIndex() << ")";
    out.setf(ios::scientific ,ios::floatfield);
    return out;
  }
}

void
Patch::performConsistencyCheck() const
{
  // make sure that the patch's size is at least [1,1,1] 
  IntVector res(getExtraCellHighIndex()-getExtraCellLowIndex());
  if(res.x() < 1 || res.y() < 1 || res.z() < 1) {
    ostringstream msg;
    msg << "Degenerate patch: " << toString() << " (resolution=" << res << ")";
    SCI_THROW(InvalidGrid( msg.str(),__FILE__,__LINE__ ));
  }
}

Patch::FaceType
Patch::stringToFaceType( const string & faceString )
{
  if(      faceString == "x+" ) {
    return xplus;
  }
  else if( faceString == "x-" ) {
    return xminus;
  }
  else if( faceString == "y+" ) {
    return yplus;
  }
  else if( faceString == "y-" ) {
    return yminus;
  }
  else if( faceString == "z+" ) {
    return zplus;
  }
  else if( faceString == "z-" ) {
    return zminus;
  }
  else {
    throw InvalidValue( "Invalid string input for stringToFaceType: '" + faceString + "'", __FILE__, __LINE__ );
  }
}

void
Patch::setBCType(Patch::FaceType face, BCType newbc)
{
  switch(face)
  {
    case xminus:
      d_patchState.xminus=static_cast<unsigned int>(newbc);
      break;
    case yminus:
      d_patchState.yminus=static_cast<unsigned int>(newbc);
      break;
    case zminus:
      d_patchState.zminus=static_cast<unsigned int>(newbc);
      break;
    case xplus:
      d_patchState.xplus=static_cast<unsigned int>(newbc);
      break;
    case yplus:
      d_patchState.yplus=static_cast<unsigned int>(newbc);
      break;
    case zplus:
      d_patchState.zplus=static_cast<unsigned int>(newbc);
      break;
    default:
      //we should throw an exception here but for some reason this doesn't compile
      throw InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
      return;
    }
}

void
Patch::printPatchBCs(ostream& out) const
{
   out << " BC types: x- " << getBCType(xminus) << ", x+ "<<getBCType(xplus)
                 << ", y- "<< getBCType(yminus) << ", y+ "<< getBCType(yplus)
                 << ", z- "<< getBCType(zminus) << ", z+ "<< getBCType(zplus)<< endl;
}

void 
Patch::setArrayBCValues( Patch::FaceType face, BCDataArray * bcda )
{
  // At this point need to set up the iterators for each BCData type:
  // Side, Rectangle, Circle, Difference, and Union.

  cout << "Patch.cc: setArrayBCValues() for face: " << face << ", id: " << d_id << "\n";

  bcda->determineIteratorLimits( face, this );

  cout << "original bcda: aaaaaaaaaaaa "<< face << " ------------------------------------------------------\n";
  bcda->print();
  cout << "AAAAAAAAAAAA------------------------------------------------------\n";

  BCDataArray * new_bcda = scinew BCDataArray( *bcda ); // FIXME... do we really need to copy this?  Why not just use the same one?

  cout << "copy of bcda: bbbbbbbbbbbb "<< face << " ------------------------------------------------------\n";
  new_bcda->print();
  cout << "BBBBBBBBBBBB------------------------------------------------------\n";

  (*d_arrayBCS)[face] = new_bcda;

  cout << "Patch.cc: BCs for face " << face << " are:\n";
  (*d_arrayBCS)[face]->print();

  // FIXME DEBUGGING:
  //cout << "fyi: d_arrayBCS for " << face << " is " << (*d_arrayBCS)[face] << "\n";
}  
 
const BCDataArray*
Patch::getBCDataArray( Patch::FaceType face ) const
{
  if( d_arrayBCS ) {
    if( (*d_arrayBCS)[face] ) {
      // FIXME DEBUGGING:
      //cout << "FYI: d_arrayBCS for " << face << " is " << (*d_arrayBCS)[face] << "\n";
      return (*d_arrayBCS)[face];
    } else {
      ostringstream msg;
      msg << "face = " << face << endl;
      SCI_THROW(InternalError("d_arrayBCS[face] has not been allocated",
                              __FILE__, __LINE__));
    }
  } else {
    SCI_THROW(InternalError("d_arrayBCS has not been allocated",
                            __FILE__, __LINE__));
  }
}

const BoundCondBase*
Patch::getArrayBCValues( Patch::FaceType   face,
                         int               mat_id,
                         const string    & type,
                         Iterator        & cell_ptr, 
                         Iterator        & node_ptr,
                         int               child ) const
{
  //cout << "Patch::getArrayBCValues() called on patch " << d_id << ", face " << face << ", matl: " << mat_id << "\n";

  BCDataArray * b0 = (*d_arrayBCS)[0];
  BCDataArray * b1 = (*d_arrayBCS)[1];
  BCDataArray * b2 = (*d_arrayBCS)[2];
  BCDataArray * b3 = (*d_arrayBCS)[3];
  BCDataArray * b4 = (*d_arrayBCS)[4];
  BCDataArray * b5 = (*d_arrayBCS)[5];

  BCDataArray* bcda = (*d_arrayBCS)[face];
  if (bcda) {
    // DEBUG print statement FIXME remove
    //bcda->print();
    // end debug
    const BoundCondBase* bc = bcda->getBoundCondData(mat_id,type,child);
    if (bc) {
      cell_ptr = bcda->getCellFaceIterator( mat_id, child, this );
      node_ptr = bcda->getNodeFaceIterator( mat_id, child, this );
    }
    return bc;
  } else {
    return 0;
  }
}

bool 
Patch::haveBC(       FaceType   face,
                     int        mat_id,
               const string   & bc_type,
               const string   & bc_variable ) const
{
  BCDataArray* itr = (*d_arrayBCS)[ face ];

  if ( itr ) {

    BCDataArray::bcDataArrayType::const_iterator v_itr = itr->d_BCDataArray.find( mat_id );

    if (v_itr != itr->d_BCDataArray.end()) { 
      for( vector<BCGeomBase*>::const_iterator it = v_itr->second.begin(); it != v_itr->second.end(); ++it ) {
        const BCData * bc = (*it)->getBCData( mat_id );
        bool found_variable = bc->exists( bc_type, bc_variable );
        if( found_variable ) {
          return true;
        }
      }
    } 
    int all_matls = -1; // -1 is the ID for all materials.

    // Check the mat_it = "all" case
    v_itr = itr->d_BCDataArray.find( all_matls );
    if (v_itr != itr->d_BCDataArray.end()) {
      for( vector<BCGeomBase*>::const_iterator it = v_itr->second.begin(); it != v_itr->second.end(); ++it ) {
        const BCData * bcd = (*it)->getBCData( mat_id );
        bool found_variable = (bcd == NULL) ? false : bcd->exists( bc_type, bc_variable );
        if (found_variable){
          return true;
        }
      }
    }
  }
  return false;
}

void
Patch::getFace(FaceType face, const IntVector& insideOffset,
               const IntVector& outsideOffset,
               IntVector& l, IntVector& h) const
{
  // don't count extra cells
  IntVector ll=getCellLowIndex();
  IntVector hh=getCellHighIndex();
  l=ll;
  h=hh;
  switch(face){
  case xminus:
    h.x(ll.x()+insideOffset.x());
    l.x(ll.x()-outsideOffset.x());
    break;
  case xplus:
    l.x(hh.x()-insideOffset.x());
    h.x(hh.x()+outsideOffset.x());
    break;
  case yminus:
    h.y(ll.y()+insideOffset.y());
    l.y(ll.y()-outsideOffset.y());
    break;
  case yplus:
    l.y(hh.y()-insideOffset.y());
    h.y(hh.y()+outsideOffset.y());
    break;
  case zminus:
    h.z(ll.z()+insideOffset.z());
    l.z(ll.z()-outsideOffset.z());
    break;
  case zplus:
    l.z(hh.z()-insideOffset.z());
    h.z(hh.z()+outsideOffset.z());
    break;
  default:
     SCI_THROW(InternalError("Illegal FaceType in Patch::getFace", __FILE__, __LINE__));
  }
}

IntVector Patch::faceDirection(FaceType face) const
{
  switch(face) {
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::faceDirection", __FILE__, __LINE__));
  }
}

string
Patch::getFaceName(FaceType face)
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::faceName", __FILE__, __LINE__));
  }
}

void
Patch::getFaceExtraNodes(FaceType face, int offset,IntVector& l,
                                                   IntVector& h) const
{
  // Change from getNodeLowIndex to getInteriorNodeLowIndex.  Need to do this
  // when we have extra cells.
  IntVector lorig=l=getExtraNodeLowIndex();
  IntVector horig=h=getExtraNodeHighIndex();
  switch(face){
  case xminus:
    l.x(lorig.x()-offset);
    h.x(lorig.x()+1-offset);
    break;
  case xplus:
    l.x(horig.x()-1+offset);
    h.x(horig.x()+offset);
    break;
  case yminus:
    l.y(lorig.y()-offset);
    h.y(lorig.y()+1-offset);
    break;
  case yplus:
    l.y(horig.y()-1+offset);
    h.y(horig.y()+offset);
    break;
  case zminus:
    l.z(lorig.z()-offset);
    h.z(lorig.z()+1-offset);
    break;
  case zplus:
    l.z(horig.z()-1+offset);
    h.z(horig.z()+offset);
    break;
  default:
    SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceExtraNodes", __FILE__, __LINE__));
  }
}

void
Patch::getFaceNodes(FaceType face, int offset,IntVector& l, IntVector& h) const
{
  // Change from getNodeLowIndex to getInteriorNodeLowIndex.  Need to do this
  // when we have extra cells.
  IntVector lorig=l=getNodeLowIndex();
  IntVector horig=h=getNodeHighIndex();
  switch(face){
  case xminus:
    l.x(lorig.x()-offset);
    h.x(lorig.x()+1-offset);
    break;
  case xplus:
    l.x(horig.x()-1+offset);
    h.x(horig.x()+offset);
    break;
  case yminus:
    l.y(lorig.y()-offset);
    h.y(lorig.y()+1-offset);
    break;
  case yplus:
    l.y(horig.y()-1+offset);
    h.y(horig.y()+offset);
    break;
  case zminus:
    l.z(lorig.z()-offset);
    h.z(lorig.z()+1-offset);
    break;
  case zplus:
    l.z(horig.z()-1+offset);
    h.z(horig.z()+offset);
    break;
  default:
    SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceNodes", __FILE__, __LINE__));
  }
}

void
Patch::getFaceCells(FaceType face, int offset,IntVector& l, IntVector& h) const
{
   IntVector lorig=l=getExtraCellLowIndex();
   IntVector horig=h=getExtraCellHighIndex();
   switch(face){
   case xminus:
      l.x(lorig.x()-offset);
      h.x(lorig.x()+1-offset);
      break;
   case xplus:
      l.x(horig.x()-1+offset);
      h.x(horig.x()+offset);
      break;
   case yminus:
      l.y(lorig.y()-offset);
      h.y(lorig.y()+1-offset);
      break;
   case yplus:
      l.y(horig.y()-1+offset);
      h.y(horig.y()+offset);
      break;
   case zminus:
      l.z(lorig.z()-offset);
      h.z(lorig.z()+1-offset);
      break;
   case zplus:
      l.z(horig.z()-1+offset);
      h.z(horig.z()+offset);
      break;
   default:
     SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceCells", __FILE__, __LINE__));
   }
}

string
Patch::toString() const
{
  char str[ 1024 ];

  Box box(getExtraBox());
  sprintf( str, "[ [%2.2f, %2.2f, %2.2f] [%2.2f, %2.2f, %2.2f] ]",
           box.lower().x(), box.lower().y(), box.lower().z(),
           box.upper().x(), box.upper().y(), box.upper().z() );

  return string( str );
}

/**
 * This function will return an iterator that touches all cells
 *  that are are partially or fully within the region formed
 *  by intersecting the box b and this patch (including extra cells).
 *  The region is inclusive on the + faces
 */
CellIterator
Patch::getCellIterator(const Box& b) const
{
   Point l = getLevel()->positionToIndex(b.lower());
   Point u = getLevel()->positionToIndex(b.upper());
   IntVector low(RoundDown(l.x()), RoundDown(l.y()), RoundDown(l.z()));
   // high is the inclusive upper bound on the index.  In order for
   // the iterator to work properly we need in increment all the
   // indices by 1.
   IntVector high(RoundDown(u.x())+1, RoundDown(u.y())+1, RoundDown(u.z())+1);
   low = Max(low, getExtraCellLowIndex());
   high = Min(high, getExtraCellHighIndex());
   return CellIterator(low, high);
}

/**
 *  This function will return an iterator that touches all cells
 *  whose centers are within the region formed
 *  by intersecting the box b and this patch (including extra cells).
 *  The region is inclusive on the + faces
 */
CellIterator
Patch::getCellCenterIterator(const Box& b) const
{
  Point l = getLevel()->positionToIndex(b.lower());
  Point u = getLevel()->positionToIndex(b.upper());
  // If we subtract 0.5 from the bounding box locations we can treat
  // the code just like we treat nodes.
  l -= Vector(0.5, 0.5, 0.5);
  u -= Vector(0.5, 0.5, 0.5);
  // This will return an empty iterator when the box is degerate.
  IntVector low(RoundUp(l.x()), RoundUp(l.y()), RoundUp(l.z()));
  IntVector high(RoundDown(u.x()) + 1, RoundDown(u.y()) + 1,
      RoundDown(u.z()) + 1);
  low = Max(low, getExtraCellLowIndex());
  high = Min(high, getExtraCellHighIndex());
  return CellIterator(low, high);
}

/*****************************************************
 * Returns a face cell iterator
 *  face specifies which face will be returned
 *  domain specifies which type of iterator will be returned
 *  and can be one of the following:
 *     ExtraMinusEdgeCells:    All extra cells beyond the face excluding the edge cells.
 *     ExtraPlusEdgeCells:     All extra cells beyond the face including the edge cells.
 *     FaceNodes:              All nodes on the face.
 *     SFCVars:                All face centered nodes on the face.
 *     InteriorFaceCells:      All cells on the interior of the face.                                                             
 */
CellIterator    
Patch::getFaceIterator(const FaceType& face, const FaceIteratorType& domain) const
{
  IntVector lowPt, highPt;

  //compute the dimension
  int dim=face/2;


  //compute if we are a plus face
  bool plusface=face%2;

  ASSERT(getBCType(face)!=Neighbor);

  switch(domain)
  {
    //start with tight fitting patch and expand the indices to include the wanted regions
    case ExtraMinusEdgeCells:
      //grab patch region without extra cells
      lowPt =  getCellLowIndex();
      highPt = getCellHighIndex();

      //select the face
      if(plusface){
          //restrict dimension to the face
          lowPt[dim]=highPt[dim];
          //extend dimension by extra cells
          highPt[dim]=getExtraCellHighIndex()[dim];
      }
      else{
          //restrict dimension to face
          highPt[dim]=lowPt[dim];
          //extend dimension by extra cells
          lowPt[dim]=getExtraCellLowIndex()[dim];
      }
      break;
      //start with the loose fitting patch and contract the indices to exclude the unwanted regions
    case ExtraPlusEdgeCells:
      //grab patch region with extra cells
      lowPt =  getExtraCellLowIndex();
      highPt = getExtraCellHighIndex();
     
      //select the face
      if(plusface){
          //move low point to plus face
          lowPt[dim]=getCellHighIndex()[dim];
      }
      else{
          //move high point to minus face
          highPt[dim]=getCellLowIndex()[dim];
      }
      break;
    case FaceNodes:
      //grab patch region without extra cells
      lowPt =  getNodeLowIndex();
      highPt = getNodeHighIndex();

      //select the face
      if(plusface){
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //extend low point by 1 cell
          lowPt[dim]=lowPt[dim]-1;
      }
      else{
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //extend high point by 1 cell
          highPt[dim]=highPt[dim]+1;
      }
      break;
    case SFCVars:
     
      //grab patch region without extra cells
      switch(dim)
      {
        case 0:
          lowPt =  getSFCXLowIndex();
          highPt = getSFCXHighIndex();
          break;

        case 1:
          lowPt =  getSFCYLowIndex();
          highPt = getSFCYHighIndex();
          break;

        case 2:
          lowPt =  getSFCZLowIndex();
          highPt = getSFCZHighIndex();
          break;
      }

      //select the face
      if(plusface){
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //extend low point by 1 cell
          lowPt[dim]=lowPt[dim]-1;
      }
      else{
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //extend high point by 1 cell
          highPt[dim]=highPt[dim]+1;
      }
      break;
    case InteriorFaceCells:
      lowPt =  getCellLowIndex();
      highPt = getCellHighIndex();
      
      //select the face
      if(plusface){
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //contract dimension by 1
          lowPt[dim]--;
      }
      else{
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //contract dimension by 1
          highPt[dim]++;
      }
      break;
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }
  return CellIterator(lowPt, highPt);
}
/**
 * Returns an iterator to the edge of two intersecting faces.
 * ExtraCells:  returns the intersecting extra cells between two faces
 * ExtraMinusCornerCells:  returns the intersecting extra cells minus the corner cells on two faces
 * SFCVars: returns the intersecting SFC vars between two faces
 */
CellIterator    
Patch::getEdgeCellIterator(const FaceType& face0, 
                           const FaceType& face1,const EdgeIteratorType &type) const
{
  FaceType face[2]={face0,face1};
  int dim[2]={getFaceDimension(face0),getFaceDimension(face1)};

  //return an empty iterator if trying to intersect the same dimension
  if(dim[0]==dim[1])
    return CellIterator(IntVector(0,0,0),IntVector(0,0,0));

  //the bounds of the patch iterators
  IntVector patchLow, patchHigh;
  IntVector patchExtraLow(0,0,0), patchExtraHigh(0,0,0);

  //determine the correct query functions
  switch(type)
  {
    case ExtraCells: case ExtraCellsMinusCorner:
      patchLow=getCellLowIndex();
      patchHigh=getCellHighIndex();
      patchExtraLow=getExtraCellLowIndex();
      patchExtraHigh=getExtraCellHighIndex();
      break;
    case ExtraSFC: case ExtraSFCMinusCorner:
      switch(dim[0])
      {
        case 0:
          patchLow=getSFCXLowIndex();
          patchHigh=getSFCXHighIndex();
          patchExtraLow=getExtraSFCXLowIndex();
          patchExtraHigh=getExtraSFCXHighIndex();
          break;
        case 1:
          patchLow=getSFCYLowIndex();
          patchHigh=getSFCYHighIndex();
          patchExtraLow=getExtraSFCYLowIndex();
          patchExtraHigh=getExtraSFCYHighIndex();
          break;
        case 2:
          patchLow=getSFCZLowIndex();
          patchHigh=getSFCZHighIndex();
          patchExtraLow=getExtraSFCZLowIndex();
          patchExtraHigh=getExtraSFCZHighIndex();
          break;
      }
      break;
    case SFC: case SFCMinusCorner:
      switch(dim[0])
      {
        case 0:
          patchLow=getSFCXLowIndex()+IntVector(1,1,1);
          patchHigh=getSFCXHighIndex()-IntVector(1,1,1);
          patchExtraLow=getSFCXLowIndex();
          patchExtraHigh=getSFCXHighIndex();
          break;
        case 1:
          patchLow=getSFCYLowIndex()+IntVector(1,1,1);
          patchHigh=getSFCYHighIndex()-IntVector(1,1,1);
          patchExtraLow=getSFCYLowIndex();
          patchExtraHigh=getSFCYHighIndex();
          break;
        case 2:
          patchLow=getSFCZLowIndex()+IntVector(1,1,1);
          patchHigh=getSFCZHighIndex()+IntVector(1,1,1);
          patchExtraLow=getSFCZLowIndex();
          patchExtraHigh=getSFCZHighIndex();
          break;
      }
       
      break;
    default:

      //set these values to quiet a compiler warning about unintialized variables
      patchLow=IntVector(0,0,0);
      patchHigh=IntVector(0,0,0);
      patchExtraLow=IntVector(0,0,0);
      patchExtraHigh=IntVector(0,0,0);

      throw SCIRun::InternalError("Invalid EdgeIteratorType Specified", __FILE__, __LINE__);
  };
  vector<IntVector>loPt(2), hiPt(2); 

  loPt[0] = loPt[1] = patchExtraLow;
  hiPt[0] = hiPt[1] = patchExtraHigh;

  //restrict to edge
  for (int f = 0; f < 2 ; f++ ) 
  {
    switch(face[f])
    {
      case xminus: case yminus: case zminus:
        //restrict to minus face
        hiPt[f][dim[f]]=patchLow[dim[f]];
        break;
      case xplus: case yplus: case zplus:
        //restrict to plus face
        loPt[f][dim[f]]=patchHigh[dim[f]];
        break;
      default:
        break;
    }

    switch(type)
    {
      //prune corner cells
      case ExtraCellsMinusCorner: case ExtraSFCMinusCorner: case SFCMinusCorner:
      {
        //compute the dimension of the face not being used
        int otherdim=3-dim[0]-dim[1];

        //remove the corner cells by pruning the other dimension's extra cells
        loPt[f][otherdim]=patchLow[otherdim];
        hiPt[f][otherdim]=patchHigh[otherdim];
        break;
      }
      default:
        //do nothing
        break;
     }
  }

  // compute the edge low and high pt from the intersection
  IntVector LowPt  = Max(loPt[0], loPt[1]);
  IntVector HighPt = Min(hiPt[0], hiPt[1]);

  return CellIterator(LowPt, HighPt);
}

Box Patch::getGhostBox(const IntVector& lowOffset,
                       const IntVector& highOffset) const
{
   return Box(getLevel()->getNodePosition(getExtraCellLowIndex()+lowOffset),
              getLevel()->getNodePosition(getExtraCellHighIndex()+highOffset));
}


/**
* This will return an iterator which will include all the nodes
* contained by the bounding box which also intersect the patch.
* If a dimension of the widget is degenerate (has a thickness of 0)
* the nearest node in that dimension is used.
*
* The patch region includes extra nodes
*/
NodeIterator
Patch::getNodeIterator(const Box& b) const
{
  // Determine if we are dealing with a 2D box.
   Point l = getLevel()->positionToIndex(b.lower());
   Point u = getLevel()->positionToIndex(b.upper());
   int low_x, low_y, low_z, high_x, high_y, high_z;
   if (l.x() != u.x()) {
     // Get the nodes that are included
     low_x = RoundUp(l.x());
     high_x = RoundDown(u.x()) + 1;
   } else {
     // Get the nodes that are nearest
     low_x = RoundDown(l.x()+0.5);
     high_x = low_x + 1;
   }
   if (l.y() != u.y()) {
     // Get the nodes that are included
     low_y = RoundUp(l.y());
     high_y = RoundDown(u.y()) + 1;
   } else {
     // Get the nodes that are nearest
     low_y = RoundDown(l.y()+0.5);
     high_y = low_y + 1;
   }
   if (l.z() != u.z()) {
     // Get the nodes that are included
     low_z = RoundUp(l.z());
     high_z = RoundDown(u.z()) + 1;
   } else {
     // Get the nodes that are nearest
     low_z = RoundDown(l.z()+0.5);
     high_z = low_z + 1;
   }
   IntVector low(low_x, low_y, low_z);
   IntVector high(high_x, high_y, high_z);
   low = Max(low, getExtraNodeLowIndex());
   high = Min(high, getExtraNodeHighIndex());
   return NodeIterator(low, high);
}

// if next to a boundary then lowIndex = 2+celllowindex in the flow dir
IntVector Patch::getSFCXFORTLowIndex__Old() const
{
  IntVector h(getFortranExtraCellLowIndex()+
              IntVector(getBCType(xminus) == Neighbor?0:2, 
                        getBCType(yminus) == Neighbor?0:1,
                        getBCType(zminus) == Neighbor?0:1));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCXFORTHighIndex__Old() const
{
   IntVector h(getFortranExtraCellHighIndex() -
               IntVector(getBCType(xplus) == Neighbor?0:1,
                         getBCType(yplus) == Neighbor?0:1,
                         getBCType(zplus) == Neighbor?0:1));
   return h;
}

// if next to a boundary then lowIndex = 2+celllowindex
IntVector Patch::getSFCYFORTLowIndex__Old() const
{
  IntVector h(getFortranExtraCellLowIndex()+
              IntVector(getBCType(xminus) == Neighbor?0:1, 
                        getBCType(yminus) == Neighbor?0:2,
                        getBCType(zminus) == Neighbor?0:1));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCYFORTHighIndex__Old() const
{
   IntVector h(getFortranExtraCellHighIndex() - 
               IntVector(getBCType(xplus) == Neighbor?0:1,
                         getBCType(yplus) == Neighbor?0:1,
                         getBCType(zplus) == Neighbor?0:1));
   return h;
}

// if next to a boundary then lowIndex = 2+celllowindex
IntVector Patch::getSFCZFORTLowIndex__Old() const
{
  IntVector h(getFortranExtraCellLowIndex()+
              IntVector(getBCType(xminus) == Neighbor?0:1, 
                        getBCType(yminus) == Neighbor?0:1,
                        getBCType(zminus) == Neighbor?0:2));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCZFORTHighIndex__Old() const
{
   IntVector h(getFortranExtraCellHighIndex() - 
               IntVector(getBCType(xplus) == Neighbor?0:1,
                         getBCType(yplus) == Neighbor?0:1,
                         getBCType(zplus) == Neighbor?0:1));
   return h;
}

/**
 * For AMR.  When there are weird patch configurations, sometimes patches can overlap.
 * Find the intersection betwen the patch and the desired dependency, and then remove the intersection.
 * If the overlap IS the intersection, set the low to be equal to the high.
 */
void Patch::cullIntersection(VariableBasis basis, IntVector bl, const Patch* neighbor,
                             IntVector& region_low, IntVector& region_high) const
{
  TAU_PROFILE("Patch::cullIntersection", " ", TAU_USER); 
  // on certain AMR grid configurations, with extra cells, patches can overlap
  // such that the extra cell of one patch overlaps a normal cell of another
  // in such conditions, we shall exclude that extra cell from MPI communication
  // Also disclude overlapping extra cells just to be safe

  // follow this heuristic - if one dimension of neighbor overlaps "this" with 2 cells
  //   then prune it back to its interior cells
  //   if two or more, throw it out entirely.  If there are 2+ different, the patches share only a line or point
  //   and the patch's boundary conditions are basically as though there were no patch there

  // use the cell-based interior to compare patch positions, but use the basis-specific one when culling the intersection
  IntVector p_int_low(getLowIndex(Patch::CellBased)), p_int_high(getHighIndex(Patch::CellBased));
  IntVector n_int_low(neighbor->getLowIndex(Patch::CellBased)), n_int_high(neighbor->getHighIndex(Patch::CellBased));

  // actual patch intersection
  IntVector diff = Abs(Max(getExtraLowIndex(Patch::CellBased, bl), neighbor->getExtraLowIndex(Patch::CellBased, bl)) -
                       Min(getExtraHighIndex(Patch::CellBased, bl), neighbor->getExtraHighIndex(Patch::CellBased, bl)));

  // go through each dimension, and determine where the neighor patch is relative to this
  // if it is above or below, clamp it to the interior of the neighbor patch
  // based on the current grid constraints, it is reasonable to assume that the patches
  // line up at least in corners.
  int bad_diffs = 0;
  for (int dim = 0; dim < 3; dim++) {
    //if the length of the side is not equal to zero,
    //is equal to 2 times the number of extra cells,
    //and the patches are adjacent on this dimension
      //then increment the bad_diffs counter
    if (diff[dim]!=0 && diff[dim] == 2*getExtraCells()[dim] 
        && (p_int_low[dim]==n_int_high[dim] || n_int_low[dim]==p_int_high[dim]) ) 
      bad_diffs++;

    // depending on the region, cull away the portion of the region that in 'this'
    if (n_int_high[dim] == p_int_low[dim]) {
      region_high[dim] = Min(region_high[dim], neighbor->getHighIndex(basis)[dim]);
    }
    else if (n_int_low[dim] == p_int_high[dim]) {
      region_low[dim] = Max(region_low[dim], neighbor->getLowIndex(basis)[dim]);
    }
  }
  
  // prune it back if heuristic met or if already empty
    //if bad_diffs is >=2 then we have a bad corner/edge that needs to be pruned
  IntVector region_diff = region_high - region_low;
  if (bad_diffs >= 2 || region_diff.x() * region_diff.y() * region_diff.z() == 0)
    region_low = region_high;  // caller will check for this case

}

void Patch::getGhostOffsets(VariableBasis basis, Ghost::GhostType gtype,
                            int numGhostCells,
                            IntVector& lowOffset, IntVector& highOffset)
{
  MALLOC_TRACE_TAG_SCOPE("Patch::getGhostOffsets");
  // This stuff works by assuming there are no neighbors.  If there are
  // neighbors, it can simply cut back appropriately later (no neighbor
  // essentially means no ghost cell on that side).
  IntVector g(numGhostCells, numGhostCells, numGhostCells);
  if (numGhostCells == 0) {
    lowOffset = highOffset = IntVector(0, 0, 0);
    return;
  }
  else if (gtype == Ghost::None)
    SCI_THROW(InternalError("ghost cells should not be specified with Ghost::None", __FILE__, __LINE__));

  if (basis == CellBased) {
    if (gtype == Ghost::AroundCells) {
      // Cells around cells
      lowOffset = highOffset = g;
    }
    else {
      // cells around nodes/faces
      IntVector aroundDir = Ghost::getGhostTypeDir(gtype);
      lowOffset = g * aroundDir;
      highOffset = lowOffset - aroundDir;
    }
  }
  else {
    // node or face based
    IntVector dir = Ghost::getGhostTypeDir((Ghost::GhostType)basis);
    if (gtype == Ghost::AroundCells) {
      // Nodes/faces around cells
      lowOffset = g - IntVector(1, 1, 1);
      highOffset = lowOffset + dir; // I think this is the right way
    }
    else if (basis == (VariableBasis)gtype) {
      // nodes around nodes or faces around faces
      lowOffset = highOffset = g * dir; 
    }
    else if (gtype == Ghost::AroundFaces) {
      lowOffset = highOffset = g;
    }
    else {
      string basisName = Ghost::getGhostTypeName((Ghost::GhostType)basis);
      string ghostTypeName = Ghost::getGhostTypeName(gtype);
      SCI_THROW(InternalError(basisName + " around " + ghostTypeName + " not supported for ghost offsets",
                              __FILE__, __LINE__));
    }
  }

  ASSERT(lowOffset[0] >= 0 && lowOffset[1] >= 0 && lowOffset[2] >= 0 &&
         highOffset[0] >= 0 && highOffset[2] >= 0 && highOffset[2] >= 0); 
}

void Patch::computeVariableExtents(VariableBasis basis,
                                   const IntVector& boundaryLayer,
                                   Ghost::GhostType gtype, int numGhostCells,
                                   IntVector& low, IntVector& high) const
{
  IntVector lowOffset, highOffset;
  getGhostOffsets(basis, gtype, numGhostCells, lowOffset, highOffset);
  computeExtents(basis, boundaryLayer, lowOffset, highOffset, low, high);
}

void Patch::computeVariableExtents(VariableBasis basis,
                                   const IntVector& boundaryLayer,
                                   Ghost::GhostType gtype, int numGhostCells,
                                   Patch::selectType& neighbors,
                                   IntVector& low, IntVector& high) const
{
  IntVector lowOffset, highOffset;
  getGhostOffsets(basis, gtype, numGhostCells, lowOffset, highOffset);
  computeExtents(basis, boundaryLayer, lowOffset, highOffset, low, high);
  getLevel()->selectPatches(low, high, neighbors);
}

void Patch::computeVariableExtents(Uintah::TypeDescription::Type basis,
                                   const IntVector& boundaryLayer,
                                   Ghost::GhostType gtype, int numGhostCells,
                                   IntVector& low, IntVector& high) const
{
  bool basisMustExist = (gtype != Ghost::None);
  computeVariableExtents(translateTypeToBasis(basis, basisMustExist),
                         boundaryLayer, gtype, numGhostCells, low, high);
}

void Patch::computeVariableExtents(Uintah::TypeDescription::Type basis,
                                   const IntVector& boundaryLayer,
                                   Ghost::GhostType gtype, int numGhostCells,
                                   Patch::selectType& neighbors,
                                   IntVector& low, IntVector& high) const
{
  bool basisMustExist = (gtype != Ghost::None);
  computeVariableExtents(translateTypeToBasis(basis, basisMustExist),
                         boundaryLayer, gtype, numGhostCells, neighbors,
                         low, high);
}

void Patch::computeExtents(VariableBasis basis,
                           const IntVector& boundaryLayer,
                           const IntVector& lowOffset,
                           const IntVector& highOffset,
                           IntVector& low, IntVector& high) const

{
  ASSERT(lowOffset[0] >= 0 && lowOffset[1] >= 0 && lowOffset[2] >= 0 &&
         highOffset[0] >= 0 && highOffset[2] >= 0 && highOffset[2] >= 0);
  
  IntVector origLowIndex = getExtraLowIndex(basis, boundaryLayer);
  IntVector origHighIndex = getExtraHighIndex(basis, boundaryLayer);
  low = origLowIndex - neighborsLow()*lowOffset;
  high = origHighIndex + neighborsHigh()*highOffset;
}

void Patch::getOtherLevelPatches(int levelOffset,
                                 Patch::selectType& selected_patches,
                                 int nPaddingCells /*=0*/) const
{
  ASSERT(levelOffset !=0);

  // include the padding cells in the final low/high indices
  IntVector pc(nPaddingCells, nPaddingCells, nPaddingCells);
  
  Point lowPt = getLevel()->getCellPosition(getExtraCellLowIndex());
  Point hiPt  = getLevel()->getCellPosition(getExtraCellHighIndex());

  const LevelP& otherLevel = getLevel()->getRelativeLevel(levelOffset);
  IntVector low  = otherLevel->getCellIndex(lowPt);
  IntVector high = otherLevel->getCellIndex(hiPt);

  if (levelOffset < 0) {
    // we don't grab enough in the high direction if the fine extra cell
    // is on the other side of a coarse boundary

    // refinement ratio between the two levels
    IntVector crr = IntVector(1,1,1);
    for (int i=1;i<=(-levelOffset);i++){
      crr = crr * otherLevel->getRelativeLevel(i)->getRefinementRatio();
    }
    IntVector highIndex = getExtraCellHighIndex();
    IntVector offset((highIndex.x() % crr.x()) == 0 ? 0 : 1,
                     (highIndex.y() % crr.y()) == 0 ? 0 : 1,
                     (highIndex.z() % crr.z()) == 0 ? 0 : 1);
    high += offset;
  }
  
  if (levelOffset > 0) {
    // the getCellPosition->getCellIndex seems to always add one...
    // maybe we should just separate this back to coarser/finer patches
    // and use mapCellToFiner...
    
    // also subtract more from low and keep high where it is to get extra 
    // cells, since selectPatches doesn't 
    // use extra cells. 
    low = low - IntVector(2,2,2);
  }

  //cout << "  Patch:Golp: " << low-pc << " " << high+pc << endl;
  Level::selectType patches;
  otherLevel->selectPatches(low-pc, high+pc, patches); 
  
  // based on the expanded range above to search for extra cells, we might
  // have grabbed more patches than we wanted, so refine them here
  for (int i = 0; i < patches.size(); i++) {
    IntVector lo = patches[i]->getExtraCellLowIndex();
    IntVector hi = patches[i]->getExtraCellHighIndex();
    bool intersect = doesIntersect(low-pc, high+pc, lo, hi );
    
    if (levelOffset < 0 || intersect) {
      selected_patches.push_back(patches[i]);
    }
  }
}
/**
 * Returns the VariableBasis for the TypeDescription::type specified
 * in type.  If mustExist is true this function will throw an exception
 * if the VariableBasis does not exist for the given type.
 */
Patch::VariableBasis Patch::translateTypeToBasis(Uintah::TypeDescription::Type type,
                                                 bool mustExist)
{
  switch(type){
  case TypeDescription::CCVariable:
    return CellBased;
  case TypeDescription::NCVariable:
    return NodeBased;
  case TypeDescription::SFCXVariable:
    return XFaceBased;
  case TypeDescription::SFCYVariable:
    return YFaceBased;
  case TypeDescription::SFCZVariable:
    return ZFaceBased;
  case TypeDescription::ParticleVariable:
  case TypeDescription::PerPatch:
    return CellBased;
  default:
    if (mustExist)
      SCI_THROW(InternalError("Unknown variable type in Patch::translateTypeToBasis",
                              __FILE__, __LINE__));
    else
      return CellBased; // doesn't matter
  }
}

/**
* Returns a Box in domain coordinates of the patch including extra cells
*/
Box Patch::getExtraBox() const {
  return getLevel()->getBox(getExtraCellLowIndex(), getExtraCellHighIndex());
}

/**
* Returns a Box in domain coordinates of the patch excluding extra cells
*/
Box Patch::getBox() const {
  return getLevel()->getBox(getCellLowIndex(),getCellHighIndex());
}

IntVector Patch::noNeighborsLow() const
{
  return IntVector(getBCType(xminus) == Neighbor? 0:1,
                   getBCType(yminus) == Neighbor? 0:1,
                   getBCType(zminus) == Neighbor? 0:1);
}

IntVector Patch::noNeighborsHigh() const
{
  return IntVector(getBCType(xplus) == Neighbor? 0:1,
                   getBCType(yplus) == Neighbor? 0:1,
                   getBCType(zplus) == Neighbor? 0:1);
}
IntVector Patch::neighborsLow() const
{
  return IntVector(getBCType(xminus) == Neighbor? 1:0,
                   getBCType(yminus) == Neighbor? 1:0,
                   getBCType(zminus) == Neighbor? 1:0);
}

IntVector Patch::neighborsHigh() const
{
  return IntVector(getBCType(xplus) == Neighbor? 1:0,
                   getBCType(yplus) == Neighbor? 1:0,
                   getBCType(zplus) == Neighbor? 1:0);
}

/**
* Returns the low index for a variable of type basis with extraCells or
* the boundaryLayer specified in boundaryLayer.  Boundary layers take 
* precidence over extra cells.
*/
IntVector Patch::getExtraLowIndex(VariableBasis basis,
                             const IntVector& boundaryLayer) const
{
  //no boundary layer so use extra cells
  if(boundaryLayer==IntVector(0,0,0))
  {
    switch (basis) {
      case CellBased:
        return getExtraCellLowIndex();
      case NodeBased:
        return getExtraNodeLowIndex();
      case XFaceBased:
        return getExtraSFCXLowIndex();
      case YFaceBased:
        return getExtraSFCYLowIndex();
      case ZFaceBased:
        return getExtraSFCZLowIndex();
      case AllFaceBased:
        SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getExtraLowIndex(basis)",
              __FILE__, __LINE__));
      default:
        SCI_THROW(InternalError("Illegal VariableBasis in Patch::getExtraLowIndex(basis)",
              __FILE__, __LINE__));
    }
  }
  else
  {
    switch (basis) {
      case CellBased:
        return getCellLowIndex()-noNeighborsLow()*boundaryLayer;
      case NodeBased:
        return getNodeLowIndex()-noNeighborsLow()*boundaryLayer;
      case XFaceBased:
        return getSFCXLowIndex()-noNeighborsLow()*boundaryLayer;
      case YFaceBased:
        return getSFCYLowIndex()-noNeighborsLow()*boundaryLayer;
      case ZFaceBased:
        return getSFCZLowIndex()-noNeighborsLow()*boundaryLayer;
      case AllFaceBased:
        SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getExtraLowIndex(basis)",
              __FILE__, __LINE__));
      default:
        SCI_THROW(InternalError("Illegal VariableBasis in Patch::getExtraLowIndex(basis)",
              __FILE__, __LINE__));
    }

  }
}
 
/**
* Returns the high index for a variable of type basis with extraCells or
* the boundaryLayer specified in boundaryLayer.  Boundary layers take precidence 
* over extra cells.
*/
IntVector Patch::getExtraHighIndex(VariableBasis basis,
                              const IntVector& boundaryLayer) const
{
 
  if(boundaryLayer==IntVector(0,0,0))
  {
    switch (basis) {
      case CellBased:
        return getExtraCellHighIndex();
      case NodeBased:
        return getExtraNodeHighIndex();
      case XFaceBased:
        return getExtraSFCXHighIndex();
      case YFaceBased:
        return getExtraSFCYHighIndex();
      case ZFaceBased:
        return getExtraSFCZHighIndex();
      case AllFaceBased:
        SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getExtraHighIndex(basis)",
              __FILE__, __LINE__));
      default:
        SCI_THROW(InternalError("Illegal VariableBasis in Patch::getExtraIndex(basis)",
              __FILE__, __LINE__));
    }
  }
  else
  {
    switch (basis) {
      case CellBased:
        return getCellHighIndex()+noNeighborsHigh()*boundaryLayer;
      case NodeBased:
        return getNodeHighIndex()+noNeighborsHigh()*boundaryLayer;
      case XFaceBased:
        return getSFCXHighIndex()+noNeighborsHigh()*boundaryLayer;
      case YFaceBased:
        return getSFCYHighIndex()+noNeighborsHigh()*boundaryLayer;
      case ZFaceBased:
        return getSFCZHighIndex()+noNeighborsHigh()*boundaryLayer;
      case AllFaceBased:
        SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getExtraHighIndex(basis)",
              __FILE__, __LINE__));
      default:
        SCI_THROW(InternalError("Illegal VariableBasis in Patch::getExtraIndex(basis)",
              __FILE__, __LINE__));
    }
  }
}

/**
* Returns the low index for a variable of type basis without a
* boundary layer and without extraCells.
*/
IntVector Patch::getLowIndex(VariableBasis basis) const
{
  switch (basis) {
  case CellBased:
    return getCellLowIndex();
  case NodeBased:
    return getNodeLowIndex();
  case XFaceBased:
    return getSFCXLowIndex();
  case YFaceBased:
    return getSFCYLowIndex();
  case ZFaceBased:
    return getSFCZLowIndex();
  case AllFaceBased:
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  }
}

/**
* Returns the high index for a variable of type basis without a
* boundary layer and without extraCells.
*/
IntVector Patch::getHighIndex(VariableBasis basis) const
{
  switch (basis) {
  case CellBased:
    return getCellHighIndex();
  case NodeBased:
    return getNodeHighIndex();
  case XFaceBased:
    return getSFCXHighIndex();
  case YFaceBased:
    return getSFCYHighIndex();
  case ZFaceBased:
    return getSFCZHighIndex();
  case AllFaceBased:
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  }
}

/**
* Returns the low index for a variable of type basis without extraCells
* except on the boundary of the domain.
*/
IntVector Patch::getLowIndexWithDomainLayer(VariableBasis basis) const
{
  IntVector inlow = getLowIndex(basis);
  IntVector low = getExtraLowIndex(basis, IntVector(0,0,0));
  if (getBCType(xminus) == None) inlow[0] = low[0];
  if (getBCType(yminus) == None) inlow[1] = low[1];
  if (getBCType(zminus) == None) inlow[2] = low[2];
  return inlow;
}

/**
* Returns the high index for a variable of type basis without extraCells
* except on the boundary of the domain.
*/
IntVector Patch::getHighIndexWithDomainLayer(VariableBasis basis) const
{
  IntVector inhigh = getHighIndex(basis);
  IntVector high = getExtraHighIndex(basis, IntVector(0,0,0));
  if (getBCType(xplus) == None) inhigh[0] = high[0];
  if (getBCType(yplus) == None) inhigh[1] = high[1];
  if (getBCType(zplus) == None) inhigh[2] = high[2];
  return inhigh;
}

void Patch::finalizePatch()
{
  TAU_PROFILE("Patch::finalizePatch()", " ", TAU_USER);
 
}

/**
 * Returns the index that this patch would be
 * if all of the levels were taken into account
 * This query is O(L) where L is the number of levels.
 */
int Patch::getGridIndex() const 
{
  int index = d_level_index;
  int levelid = getLevel()->getIndex();
  GridP grid = getLevel()->getGrid();

  // add up all the patches in the preceding levels
  for ( int i = 0; i < levelid && i < grid->numLevels(); i++) {
    index += grid->getLevel(i)->numPatches();
  }
  return index;

}


/**
* sets the vector cells equal to the list of cells that at the intersection of three faces extra cells
*/
void Patch::getCornerCells(vector<IntVector> & cells, const FaceType& face) const
{
  //set bounds for loops below
  int xstart=0,xend=2;
  int ystart=0,yend=2;
  int zstart=0,zend=2;

  //restrict one of the dimensions to the minus or plus face
  switch(face)
  {
    //if a minus face lower the end point (skip the plus face)
    case xminus:
      xend--;
      break;
    case yminus:
      yend--;
      break;
    case zminus:
      zend--;
      break;
    //if a plus face raise the begining point (skip the minus face)
    case xplus:
      xstart++;
      break;
    case yplus:
      ystart++;
      break;
    case zplus:
      zstart++;
      break;
    default:
      throw InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
      break;

  }

  //2D array of indices first dimension is determined by minus or plus face
  //second dimension is the low point/high point of the corner
  IntVector indices[2][2]={ 
                            {getExtraCellLowIndex(),getCellLowIndex()},  //x-,y-,z- corner
                            {getCellHighIndex(),getExtraCellHighIndex()} //x+,y+,z+ corner
                          };

  //these arrays store if the face is a neighbor or not
  bool xneighbor[2]={getBCType(xminus)==Neighbor,getBCType(xplus)==Neighbor};
  bool yneighbor[2]={getBCType(yminus)==Neighbor,getBCType(yplus)==Neighbor};
  bool zneighbor[2]={getBCType(zminus)==Neighbor,getBCType(zplus)==Neighbor};

  cells.clear();
 
  for(int x=xstart;x<xend;x++)  //For the x faces
  {
    if(xneighbor[x])  //if we have a neighbor on this face a corner cannot exist
      continue;         //continue to next x face
    
    for(int y=ystart;y<yend;y++)  //For the y faces
    {
      if(yneighbor[y]) //if we have a neighbor on this face a corner cannot exist
        continue;         //continue to next y face
      
      for(int z=zstart;z<zend;z++) //For the z faces
      {
        if(zneighbor[z]) //if we have a neighbor on this face a corner cannot exist
          continue;         //continue to next z face 
        
        //a corner exists
    
        //grab low and high points of corner from indices array
        IntVector low=IntVector(indices[x][0].x(),indices[y][0].y(),indices[z][0].z());
        IntVector high=IntVector(indices[x][1].x(),indices[y][1].y(),indices[z][1].z());

        //add corner cells to the cells vector
        for(CellIterator iter(low,high);!iter.done();iter++)
          cells.push_back(*iter);

      } // end z face loop
    } // end y face loop
  } //end x face loop
}

void Patch::initializeBoundaryConditions()
{
  if (d_arrayBCS) {
    for (unsigned int i = 0; i< 6; ++i) {
      delete (*d_arrayBCS)[i];
    }
    d_arrayBCS->clear();
    delete d_arrayBCS;
  }
  d_arrayBCS = scinew vector<BCDataArray*>(6);
  for (unsigned int i = 0; i< 6; ++i)
    (*d_arrayBCS)[i] = 0;
}

//__________________________________
//  Returns a vector of Regions that
// do not have any overlapping finer level cells.
// Use this if you want to iterate over the "finest" level cells
// on coarse and fine levels.
//
//  usage:
//  vector<Region> regions
//  coarsePatch->getFinestRegionsOnPatch(regions)
//
//  for(vector<Region>::iterator region=regions.begin();region!=regions.end();region++){
//    for (CellIterator iter(region->getLow(), region->getHigh()); !iter.done(); iter++){
//    }
//  }
void Patch::getFinestRegionsOnPatch(vector<Region>& difference) const
{
  const Level* level = getLevel();
  vector<Region> coarsePatch_q,finePatch_q;                                              
  IntVector zero(0,0,0);
  finePatch_q.push_back(Region(zero,zero));
  
  //only search for fine patches if the finer level exists
  if(level->hasFinerLevel()){ 
    Patch::selectType finePatches;
    getFineLevelPatches(finePatches); 
    const LevelP& fineLevel = getLevel()->getFinerLevel(); 
    
    //add overlapping fine patches to finePatch_q                                                       
    for(int fp=0;fp<finePatches.size();fp++){
      IntVector lo = fineLevel->mapCellToCoarser(finePatches[fp]->getCellLowIndex() );
      IntVector hi = fineLevel->mapCellToCoarser(finePatches[fp]->getCellHighIndex());                              
      finePatch_q.push_back(Region(lo, hi));                        
    }                                                           
  }                                                                                                   

  //add coarse patch to coarsePatch_q                                                                 
  coarsePatch_q.push_back(Region(getCellLowIndex(),getCellHighIndex())); 

  //compute region of coarse patches that do not contain fine patches                                 
  difference=Region::difference(coarsePatch_q, finePatch_q);                                                                                 
}

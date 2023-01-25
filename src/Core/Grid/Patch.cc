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

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Math/Primes.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/MasterLock.h>

#include <atomic>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <map>

using namespace std;
using namespace Uintah;


static std::atomic<int32_t>  ids{0};
static Uintah::MasterLock    ids_init{};

extern Uintah::MasterLock coutLock; // Used to sync cout when output by multiple ranks

Patch::Patch( const Level        * level
            , const IntVector    & lowIndex
            , const IntVector    & highIndex
            , const IntVector    & inLowIndex
            , const IntVector    & inHighIndex
            ,       unsigned int   levelIndex
            ,       int            id
            )
  : d_lowIndex(inLowIndex)
  , d_highIndex(inHighIndex)
  , d_id(id)
{

  if (d_id == -1) {
    d_id = ids.fetch_add(1, memory_order_relaxed);
  }
  else {
    if (d_id >= ids) {
      ids.store(d_id + 1, memory_order_relaxed);
    }
  }

  // DON'T call setBCType here     
  d_patchState.xminus = None;
  d_patchState.yminus = None;
  d_patchState.zminus = None;
  d_patchState.xplus  = None;
  d_patchState.yplus  = None;
  d_patchState.zplus  = None;

  //set the level index
  d_patchState.levelIndex = levelIndex;

}

Patch::Patch(const Patch* realPatch, const IntVector& virtualOffset)
    : 
      d_lowIndex(realPatch->getCellLowIndex()+virtualOffset),
      d_highIndex(realPatch->getCellHighIndex()+virtualOffset),
      d_grid(realPatch->d_grid),
      d_realPatch(realPatch), d_level_index(realPatch->d_level_index),
      d_arrayBCS(realPatch->d_arrayBCS),
      d_interiorBndArrayBCS(realPatch->d_interiorBndArrayBCS)
{
  // make the id be -1000 * realPatch id - some first come, first serve index
  d_id = -1000 * realPatch->d_id; // temporary
  int index = 1;
  int numVirtualPatches = 0;
  
  //set the level index
  d_patchState.levelIndex=realPatch->d_patchState.levelIndex;

  for (Level::const_patch_iterator iter = getLevel()->allPatchesBegin(); iter != getLevel()->allPatchesEnd(); iter++) {
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
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)) {
    if ( d_arrayBCS)
      delete (*d_arrayBCS)[face];
    
    if (d_interiorBndArrayBCS) {
      delete (*d_interiorBndArrayBCS)[face];
    }
  }

  if (d_arrayBCS) {
    d_arrayBCS->clear();
    delete d_arrayBCS;
  }

  if (d_interiorBndArrayBCS) {
    d_interiorBndArrayBCS->clear();
    delete d_interiorBndArrayBCS;
  }
}

/**
* Returns the 8 nodes found around the point pos
*/
void Patch::findCellNodes(const Point& pos, IntVector ni[8]) const
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
void Patch::findCellNodes27(const Point& pos, IntVector ni[27]) const
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
  ni[16] = IntVector(ix+1,  iy+nny,  iz+1);
  ni[17] = IntVector(ix+nnx,iy+nny,  iz+1);
  ni[18] = IntVector(ix,    iy,      iz+nnz);
  ni[19] = IntVector(ix+1,  iy,      iz+nnz);
  ni[20] = IntVector(ix+nnx,iy,      iz+nnz);
  ni[21] = IntVector(ix,    iy+1,    iz+nnz);
  ni[22] = IntVector(ix+1,  iy+1,    iz+nnz);
  ni[23] = IntVector(ix+nnx,iy+1,    iz+nnz);
  ni[24] = IntVector(ix,    iy+nny,  iz+nnz);
  ni[25] = IntVector(ix+1,  iy+nny,  iz+nnz);
  ni[26] = IntVector(ix+nnx,iy+nny,  iz+nnz);
}


/**
 *  Allows a component to add a boundary condition, if it doesn't already exist.
 */
void
Patch::possiblyAddBC( const Patch::FaceType face,             // face
                      const int             child,            // child (each child is only applicable to one face)
                      const string        & desc,             // new field label (label) 
                            int             mat_id,           // material 
                      const double          bc_value,         // value of boundary condition
                      const string        & bc_kind,          // bc type, dirichlet or neumann
                      const string        & bcFieldName,      // identifier field variable Name (var)
                      const string        & faceName ) const  //  
{
    // avoid adding duplicate boundary conditions 
  if (getModifiableBCDataArray(face)->checkForBoundCondData(mat_id,bcFieldName,child)  ){  // avoid adding duplicate boundary conditions 
    return;
  }
  if (getModifiableBCDataArray(face)->checkForBoundCondData(mat_id,desc,child)  ){  // avoid seg fault, when there are no boundary conditions on a face 

    if ( getModifiableBCDataArray(face)->getBCGeom(mat_id)[child]->getBCName()  == faceName  ){
      BoundCondBase* bc;
      BoundCondFactory::customBC( bc, mat_id, faceName, bc_value,bcFieldName, bc_kind );
      getModifiableBCDataArray(face)->getBCGeom(mat_id)[child]->sudoAddBC(bc);
      delete bc;
    }
  }
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
                               IntVector    cellIndex[8] )
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
    coutLock.lock();  // needed to eliminate threadsanitizer warnings
    out.setf(ios::scientific,ios::floatfield);
    out.precision(4);
    out << "(Patch " << r.getID() 
        << ", lowIndex=" << r.getExtraCellLowIndex() << ", highIndex=" 
        << r.getExtraCellHighIndex() << ")";
    out.setf(ios::scientific ,ios::floatfield);
    coutLock.unlock();
    return out;
  }
    
  ostream&
  operator<<(ostream& out, const Patch::FaceType& face)
  {
    out << Patch::getFaceName(face);
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
  out << " BC types: x- " << getBCType(xminus) << ", x+ "<< getBCType(xplus)
                << ", y- "<< getBCType(yminus) << ", y+ "<< getBCType(yplus)
                << ", z- "<< getBCType(zminus) << ", z+ "<< getBCType(zplus) << endl;
}

//-----------------------------------------------------------------------------------------------

void
Patch::setArrayBCValues( Patch::FaceType face, BCDataArray * bc )
{
  // At this point need to set up the iterators for each BCData type:
  // Side, Rectangle, Circle, Difference, and Union.
  BCDataArray* bctmp = bc->clone();
  bctmp->determineIteratorLimits(face,this);
  (*d_arrayBCS)[face] = bctmp->clone();
  delete bctmp;
}  

//-----------------------------------------------------------------------------------------------

void
Patch::setInteriorBndArrayBCValues(Patch::FaceType face, BCDataArray* bc)
{
  // At this point need to set up the iterators for each BCData type:
  // Side, Rectangle, Circle, Difference, and Union.
  BCDataArray* bctmp = bc->clone();
  bctmp->determineInteriorBndIteratorLimits(face,this);
  (*d_interiorBndArrayBCS)[face] = bctmp->clone();
  delete bctmp;
}

//-----------------------------------------------------------------------------------------------

const BCDataArray* Patch::getBCDataArray(Patch::FaceType face) const
{
  if (d_arrayBCS) {
    if ((*d_arrayBCS)[face]) {
      return (*d_arrayBCS)[face];
    } else {
      ostringstream msg;
      msg << "face = " << face << endl;
      SCI_THROW(InternalError("d_arrayBCS[face] has not been allocated",
                              __FILE__, __LINE__));
    }
  } else {
    SCI_THROW(InternalError("Error: d_arrayBCs not allocated. This means that no boundary conditions were found. If you are solving a periodic problem, please add a <periodic> tag to your input file to avoid this error. Otherwise, add a <BoundaryConditions> block.",
                            __FILE__, __LINE__));
  }

}

    /**
     *  \author  Derek Harris
     *  \date    September, 2015
     *  Allows a component to alter or add a boundary condition.  
     */
BCDataArray* Patch::getModifiableBCDataArray(Patch::FaceType face) const
{
  if (d_arrayBCS) {
    if ((*d_arrayBCS)[face]) {
      return (*d_arrayBCS)[face];
    } else {
      ostringstream msg;
      msg << "face = " << face << endl;
      SCI_THROW(InternalError("d_arrayBCS[face] has not been allocated",
                              __FILE__, __LINE__));
    }
  } else {
    SCI_THROW(InternalError("Error: d_arrayBCs not allocated. This means that no boundary conditions were found. If you are solving a periodic problem, please add a <periodic> tag to your input file to avoid this error. Otherwise, add a <BoundaryConditions> block.",
                            __FILE__, __LINE__));
  }

}

//-----------------------------------------------------------------------------------------------

const BCDataArray* Patch::getInteriorBndBCDataArray(Patch::FaceType face) const
{
  if (d_interiorBndArrayBCS) {
    if ((*d_interiorBndArrayBCS)[face]) {
      return (*d_interiorBndArrayBCS)[face];
    } else {
      ostringstream msg;
      msg << "face = " << face << endl;
      SCI_THROW(InternalError("d_interiorBndArrayBCS[face] has not been allocated",
                              __FILE__, __LINE__));
    }
  } else {
    SCI_THROW(InternalError("Error: d_interiorBndArrayBCS not allocated. This means that no boundary conditions were found. If you are solving a periodic problem, please add a <periodic> tag to your input file to avoid this error. Otherwise, add a <BoundaryConditions> block.",
                            __FILE__, __LINE__));
  }
  
}

//-----------------------------------------------------------------------------------------------

const BoundCondBase*
Patch::getArrayBCValues(Patch::FaceType face,
                        int mat_id,
                        const string& type,
                        Iterator& cell_ptr, 
                        Iterator& node_ptr,
                        int child) const
{
  BCDataArray* bcd = (*d_arrayBCS)[face];
  if (bcd) {
    bcd->print();
    const BoundCondBase* bc = bcd->getBoundCondData(mat_id,type,child);
    if (bc) {
      bcd->getCellFaceIterator(mat_id,cell_ptr,child);
      bcd->getNodeFaceIterator(mat_id,node_ptr,child);
    }
    return bc;
  } else {
    return 0;
  }
}

//-----------------------------------------------------------------------------------------------

const BoundCondBase*
Patch::getInteriorBndArrayBCValues(Patch::FaceType face,
                        int mat_id,
                        const string& type,
                        Iterator& cell_ptr,
                        Iterator& node_ptr,
                        int child) const
{
  BCDataArray* bcd = (*d_interiorBndArrayBCS)[face];
  if (bcd) {
    bcd->print();
    const BoundCondBase* bc = bcd->getBoundCondData(mat_id,type,child);
    if (bc) {
      bcd->getCellFaceIterator(mat_id,cell_ptr,child);
      bcd->getNodeFaceIterator(mat_id,node_ptr,child);
    }
    return bc;
  } else {
    return 0;
  }
}

//-----------------------------------------------------------------------------------------------

bool 
Patch::haveBC(FaceType face,int mat_id,const string& bc_type,
              const string& bc_variable) const
{
  BCDataArray* itr = (*d_arrayBCS)[face];

  if ( itr ) {
#if 0
    cout << "Inside haveBC" << endl;
    ubc->print();
#endif
    BCDataArray::bcDataArrayType::const_iterator v_itr;
    vector<BCGeomBase*>::const_iterator it;
    
    v_itr = itr->d_BCDataArray.find(mat_id);
    if (v_itr != itr->d_BCDataArray.end()) { 
      for (it = v_itr->second.begin(); it != v_itr->second.end(); ++it) {
        BCData bc;
        (*it)->getBCData(bc);
        bool found_variable = bc.find(bc_type,bc_variable);
        if (found_variable){
          return true;
        }
      }
    } 
    // Check the mat_it = "all" case
    v_itr = itr->d_BCDataArray.find(-1);
    if (v_itr != itr->d_BCDataArray.end()) {
      for (it = v_itr->second.begin(); it != v_itr->second.end(); ++it) {
        BCData bc;
        (*it)->getBCData(bc);
        bool found_variable = bc.find(bc_type,bc_variable);
        if (found_variable){
          return true;
        }
      }
    }
  }
  return false;
}

//-----------------------------------------------------------------------------------------------

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

      throw Uintah::InternalError("Invalid EdgeIteratorType Specified", __FILE__, __LINE__);
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

//______________________________________________________________________
/*
 * Needed for AMR inside corner patch configurations where patches can overlap.
 * Find the intersection between the patches and remove the intersecting extra cells.
 * If the overlap IS the intersection, set the low to be equal to the high.
 */
void Patch::cullIntersection(VariableBasis basis, 
                             IntVector bl, 
                             const Patch* neighbor,
                             IntVector& region_low, 
                             IntVector& region_high) const
{
  /*
   On inside corner patch configurations, with extra cells, patches can overlap and the
   extra cells of one patch overlap the normal cell of another patch.  This method removes the
   overlapping extra cells.

   If a dimension of a neighbor's patch overlaps this patch by extra cells then prune it back
   to its interior cells. If two or more cells overlap throw it out entirely.  If there are 2+
   different, the patches share only a line or point and the patch's boundary conditions are
   basically as though there were no patch there
   */

  if( neighbor == this ){
    return;
  }
  // use the cell-based interior to compare patch positions, but use the basis-specific one when culling the intersection
  IntVector p_int_low(  getLowIndex(Patch::CellBased) );
  IntVector p_int_high( getHighIndex(Patch::CellBased) );
  IntVector n_int_low(  neighbor->getLowIndex(Patch::CellBased) );
  IntVector n_int_high( neighbor->getHighIndex(Patch::CellBased) );

  // actual patch intersection
  IntVector overlapCells = Abs( Max( getExtraLowIndex(Patch::CellBased, bl), neighbor->getExtraLowIndex(Patch::CellBased, bl) ) -
                                Min( getExtraHighIndex(Patch::CellBased, bl), neighbor->getExtraHighIndex(Patch::CellBased, bl) ) );

  // Go through each dimension and determine if there is overlap.
  // Clamp it to the interior of the neighbor patch based.  It is
  // assumed that the patches line up at least in corners.
  int counter = 0;
  
  for (int dim = 0; dim < 3; dim++) {
    // If the number of overlapping cells is a) not equal to zero, b) is equal to 2 
    // extra cells, and c) the patches are adjacent on this dimension then increment counter.

    if ( overlapCells[dim] != 0 && overlapCells[dim] == 2*getExtraCells()[dim] && ( p_int_low[dim] == n_int_high[dim] || n_int_low[dim] == p_int_high[dim] ) ) {
      counter++;
    }

    // take the intersection
    if (n_int_high[dim] == p_int_low[dim]) {
      region_high[dim] = Min( region_high[dim], neighbor->getHighIndex(basis)[dim] );
    }
    else if (n_int_low[dim] == p_int_high[dim]) {
      region_low[dim] = Max( region_low[dim], neighbor->getLowIndex(basis)[dim] );
    }
  }
  
  // if counter is >=2 then we have a bad corner/edge 
  IntVector region_diff = region_high - region_low;
  int nRegionCells = region_diff.x() * region_diff.y() * region_diff.z();
  if (counter >= 2 || nRegionCells == 0){
    region_low = region_high;  // caller will check for this case
  }

}
//______________________________________________________________________
//
void Patch::getGhostOffsets(VariableBasis basis, Ghost::GhostType gtype,
                            int numGhostCells,
                            IntVector& lowOffset, IntVector& highOffset)
{
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
      // cells around nodes/faces
    if (gtype == Ghost::AroundCells) {
      lowOffset = highOffset = g;
    }else if (gtype > Ghost::AroundFaces  ){
      IntVector aroundDir = Ghost::getGhostTypeDir(gtype);
      lowOffset = g * (IntVector(1,1,1)-aroundDir);
      highOffset =g * aroundDir;
    }else {
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


/*______________________________________________________________________

                                                    +-------+-------+
                                                    |       |       |  Level Hi
                                        +***************14*****+15  |
                                        *           |       |  *    |
ExtraCells not drawn             Region of interest +----------*----+
                                        *           |       |  *    |
                                        *    void   |   12  |  *13  |
                                        *           |       |  *    |
                    +-------+-------+---*---+------------------*----+
                    |       |       |   *   |       |       |  *    |
                    |   6   |   7   |   +*******9*******10*****+11  |
                    |       |       |       |       |       |       |
                    +-----------------------------------------------+
                    |       |       |       |       |       |       |
        Patches     |   0   |   1   |   2   |   3   |   4   |   5   |
                    |       |       |       |       |       |       |
                    +-------+-------+-------+-------+-------+-------+
                    
This will return the low and high cell index of the region of interest in non-cubic computational domains.
Note the low and high point encompasses the void region show above.  This just
adds a clamp so the low and high indices don't exceed the level's extents.
This is almost identical to the patch::computeExtents
               
//______________________________________________________________________*/
void Patch::computeVariableExtentsWithBoundaryCheck(Uintah::TypeDescription::Type basis,
                                                    const IntVector& boundaryLayer,
                                                    Ghost::GhostType gtype, 
                                                    int numGhostCells,
                                                    IntVector& low, 
                                                    IntVector& high) const
{
  // This ignores virtual patches because we don't want to "clamp" this
  // extents of periodic boundary conditions to the level's extents.
 
  if ( getLevel()->isNonCubic() && numGhostCells >=1 && !isVirtual()) {

    bool basisMustExist = (gtype != Ghost::None);
    VariableBasis vbasis = translateTypeToBasis(basis, basisMustExist); 

    low  = getExtraLowIndex(vbasis, boundaryLayer);
    high = getExtraHighIndex(vbasis, boundaryLayer);
 
    //__________________________________
    // adjust the variable extents to include ghost cells EVEN when there are 
    // no neighboring patches.  This will extend outside the domain's extents,
    // which is OK. This is the key difference between computeExtents.
    IntVector ghostLowOffset, ghostHighOffset;    
    getGhostOffsets(basis, gtype, numGhostCells, ghostLowOffset, ghostHighOffset);

    IntVector ghostLow  = low  - ghostLowOffset;
    IntVector ghostHigh = high + ghostHighOffset;
                                         
    //__________________________________
    // get extents over entire level including extra cells
    
    //TODO: getLevel()->computeVariableExtents doesn't return extra cells for
    //NCVariables, so this receives back incorrect levelLow and levelHigh. (My guess is
    //that Level.cc's computeVariableExtents shouldn't be ignoring extra cells as 
    //it currently does).  Brad P. - 10/13/16
    IntVector levelLow, levelHigh;
    
    this->getLevel()->computeVariableExtents(basis, levelLow, levelHigh);
    
    // scjmc: we need to add ghost cells to level extents as well in order to
    // handle periodic boundaries on non cubic levels (such as amr fine levels)
    levelLow  -= ghostLowOffset;
    levelHigh += ghostHighOffset;
    
    //__________________________________
    //  Clamp the a valid extent
    low  = Uintah::Max( ghostLow, levelLow);
    high = Uintah::Min( ghostHigh, levelHigh);
  }
  else {
    // Do it the usual way 
    computeVariableExtents( basis, boundaryLayer, gtype, numGhostCells, low, high);
  } 
}

//______________________________________________________________________
// d
void Patch::getOtherLevelPatches(int levelOffset,
                                 Patch::selectType& selected_patches,
                                 int nPaddingCells /*=0*/) const
{
  ASSERT(levelOffset !=0);

  // include the padding cells in the final low/high indices
  IntVector pc(nPaddingCells, nPaddingCells, nPaddingCells);
  const LevelP& otherLevel = getLevel()->getRelativeLevel(levelOffset);
  Level::selectType patches;
  IntVector low(-9,-9,-9);
  IntVector high(-9,-9,-9);

  if (levelOffset < 0) {
    Point lowPt = getLevel()->getCellPosition(getExtraCellLowIndex());
    Point hiPt  = getLevel()->getCellPosition(getExtraCellHighIndex());
    low  = otherLevel->getCellIndex(lowPt);
    high = otherLevel->getCellIndex(hiPt);

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
  }else if (levelOffset > 0) {
    //Going from coarse to fine.
    Point lowPt = getLevel()->getNodePosition(getExtraCellLowIndex()); //getNodePosition is a way to get us the bottom/left/close corner coordinate.
    Point hiPt = getLevel()->getNodePosition(getExtraCellHighIndex()); //Need to add (1,1,1) to get the upper/right/far corner coordinate.
    low  = otherLevel->getCellIndex(lowPt);
    high = otherLevel->getCellIndex(hiPt);
    //Note, if extra cells were used, then the computed low and high will go too far.
    //For example, a coarse to fine refinement ratio of 2 means that if the coarse had 1 layer of extra cells,
    //then trying to find the low from the coarses (-1, -1, -1) will result in the answer (-2,-2,-2).
    //That's fine, it's a perfect projection of what it would be.
  }

  //std::cout << "  Patch:Golp: " << low-pc << " " << high+pc << std::endl;
  otherLevel->selectPatches(low-pc, high+pc, patches); 
  
  // based on the expanded range above to search for extra cells, we might
  // have grabbed more patches than we wanted, so refine them here
  for (size_t i = 0; i < patches.size(); i++) {
    IntVector lo = patches[i]->getExtraCellLowIndex();
    IntVector hi = patches[i]->getExtraCellHighIndex();
    bool intersect = doesIntersect(low-pc, high+pc, lo, hi );
    if (levelOffset < 0 || intersect) {
      selected_patches.push_back(patches[i]);
    }
  }
}

// This is being put in until the other getOtherLevelPatches can be
// made to work with both AMRMPM and Arches
void Patch::getOtherLevelPatches55902(int levelOffset,
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
  for (size_t i = 0; i < patches.size(); i++) {
    IntVector lo = patches[i]->getExtraCellLowIndex();
    IntVector hi = patches[i]->getExtraCellHighIndex();
    bool intersect = doesIntersect(low-pc, high+pc, lo, hi );
    
    if (levelOffset < 0 || intersect) {
      selected_patches.push_back(patches[i]);
    }
  }
}

void Patch::getOtherLevelPatchesNB(int levelOffset,
                                   Patch::selectType& selected_patches,
                                   int nPaddingCells /*=0*/) const
{
  ASSERT(levelOffset !=0);

  // include the padding cells in the final low/high indices
  IntVector pc(nPaddingCells, nPaddingCells, nPaddingCells);
  
  Point lowPt = getLevel()->getNodePosition(getExtraNodeLowIndex());
  Point hiPt  = getLevel()->getNodePosition(getExtraNodeHighIndex());

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
    IntVector highIndex = getExtraNodeHighIndex();
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
  for (size_t i = 0; i < patches.size(); i++) {
    IntVector lo = patches[i]->getExtraNodeLowIndex();
    IntVector hi = patches[i]->getExtraNodeHighIndex();
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
  case TypeDescription::SoleVariable:
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
* precedence over extra cells.
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
* the boundaryLayer specified in boundaryLayer.  Boundary layers take precedence 
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

//-----------------------------------------------------------------------------------------------

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
  
  if (d_interiorBndArrayBCS) {
    for (unsigned int i = 0; i< 6; ++i) {
      delete (*d_interiorBndArrayBCS)[i];
    }
    d_interiorBndArrayBCS->clear();
    delete d_interiorBndArrayBCS;
  }
  d_interiorBndArrayBCS = scinew vector<BCDataArray*>(6);
  for (unsigned int i = 0; i< 6; ++i)
    (*d_interiorBndArrayBCS)[i] = 0;
}

//-----------------------------------------------------------------------------------------------

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
    for(size_t fp=0;fp<finePatches.size();fp++){
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

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Math/Primes.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Containers/StaticArray.h>
#include <TauProfilerForSCIRun.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <map>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

static AtomicCounter ids("Patch ID counter",0);
static Mutex ids_init("ID init");
IntVector Patch::d_extraCells;
GridP Patch::d_grid[2]={0,0};
int Patch::d_newGridIndex=0;


Patch::Patch(const Level* level,
	     const IntVector& lowIndex, const IntVector& highIndex,
	     const IntVector& inLowIndex, const IntVector& inHighIndex, 
       unsigned int levelIndex,  int id)
    : d_lowIndex__New(inLowIndex), d_highIndex__New(inHighIndex), 
    
      d_realPatch(0),d_level(level),  d_level_index(-1),
      d_lowIndex(lowIndex),d_highIndex(highIndex),
      d_inLowIndex(inLowIndex), d_inHighIndex(inHighIndex),
      d_id( id )
{
  
  if(d_id == -1){
    d_id = ids++;

  } else {
    if(d_id >= ids)
      ids.set(d_id+1);
  }
   
  // DON'T call setBCType here     
  d_patchState.xminus=None;
  d_patchState.yminus=None;
  d_patchState.zminus=None;
  d_patchState.xplus=None;
  d_patchState.yplus=None;
  d_patchState.zplus=None;

  //set the level index
  d_patchState.levelIndex=levelIndex;

  d_nodeHighIndex = d_highIndex+
    IntVector(getBCType(xplus) == Neighbor?0:1,
	      getBCType(yplus) == Neighbor?0:1,
	      getBCType(zplus) == Neighbor?0:1);
  
}

Patch::Patch(const Patch* realPatch, const IntVector& virtualOffset)
    : 
      d_lowIndex__New(realPatch->d_inLowIndex+virtualOffset),
      d_highIndex__New(realPatch->d_inHighIndex+virtualOffset),
      

      d_realPatch(realPatch), d_level(realPatch->d_level),
      d_level_index(realPatch->d_level_index),
      d_lowIndex(realPatch->d_lowIndex + virtualOffset), 
      d_highIndex(realPatch->d_highIndex + virtualOffset),

      d_inLowIndex(realPatch->d_inLowIndex + virtualOffset),
      d_inHighIndex(realPatch->d_inHighIndex + virtualOffset),
      d_nodeHighIndex(realPatch->d_nodeHighIndex + virtualOffset),
      array_bcs(realPatch->array_bcs)
{
  //if(!ids){
  // make the id be -1000 * realPatch id - some first come, first serve index
  d_id = -1000 * realPatch->d_id; // temporary
  int index = 1;
  int numVirtualPatches = 0;

  for (Level::const_patchIterator iter = d_level->allPatchesBegin(); iter != d_level->allPatchesEnd(); iter++) {
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
  
  //set the level index
  d_patchState.levelIndex=realPatch->d_patchState.levelIndex;
}

Patch::~Patch()
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)) {
    delete array_bcs[face];
  }
  array_bcs.clear();
}

/**
* Returns the 8 nodes found around the point pos
*/
void Patch::findCellNodes(const Point& pos, IntVector ni[8]) const
{
  Point cellpos = getLevel__New()->positionToIndex(pos);
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
  Point cellpos = getLevel__New()->positionToIndex(pos);
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


Point Patch::nodePosition(const IntVector& idx) const {
  return d_level->getNodePosition(idx);
}

Point Patch::cellPosition(const IntVector& idx) const {
  return d_level->getCellPosition(idx);
}

void Patch::findCellsFromNode( const IntVector& nodeIndex,
                               IntVector cellIndex[8]) 
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

void Patch::findNodesFromCell( const IntVector& cellIndex,
                               IntVector nodeIndex[8])
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
    out << "(Patch " << r.getID() << ": box=" << r.getBox()
        << ", lowIndex=" << r.getCellLowIndex() << ", highIndex=" 
        << r.getCellHighIndex() << ")";
    out.setf(ios::scientific ,ios::floatfield);
    return out;
  }
}

long Patch::totalCells() const
{
   IntVector res(d_highIndex-d_lowIndex);
   return res.x()*res.y()*res.z();
}

void
Patch::performConsistencyCheck() const
{
  // make sure that the patch's size is at least [1,1,1] 
  IntVector res(d_highIndex-d_lowIndex);
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
      d_patchState.xminus=newbc;
      break;
    case yminus:
      d_patchState.yminus=newbc;
      break;
    case zminus:
      d_patchState.zminus=newbc;
      break;
    case xplus:
      d_patchState.xplus=newbc;
      break;
    case yplus:
      d_patchState.yplus=newbc;
      break;
    case zplus:
      d_patchState.zplus=newbc;
      break;
    default:
      //we should throw an exception here but for some reason this doesn't compile
      throw InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
      return;
    }

   if (newbc != Patch::Neighbor ) {
     // assign patch's extra cells here (doesn't happen in 
     // Grid::problemSetup (for coarse boundaries anyway) and helps out the regridder
     bool low = face == xminus || face == yminus || face == zminus;
     int dim = face / 2;  // do this to not have to have 6 if/else statements
     int ec = getLevel()->getExtraCells()[dim];
     if (low) { 
       this->d_lowIndex[dim] = this->d_inLowIndex[dim] - ec;
     }
     else {
       this->d_highIndex[dim] = this->d_inHighIndex[dim] + ec;
     }
   }

        
   d_nodeHighIndex = d_highIndex + IntVector(getBCType(xplus) == Neighbor?0:1,
                                             getBCType(yplus) == Neighbor?0:1,
                                             getBCType(zplus) == Neighbor?0:1);
}

void
Patch::printPatchBCs(ostream& out) const
{
   out << " BC types: x- " << getBCType(xminus) << ", x+ "<<getBCType(xplus)
                 << ", y- "<< getBCType(yminus) << ", y+ "<< getBCType(yplus)
                 << ", z- "<< getBCType(zminus) << ", z+ "<< getBCType(zplus)<< endl;
}

void 
Patch::setArrayBCValues(Patch::FaceType face, BCDataArray* bc)
{
  // At this point need to set up the iterators for each BCData type:
  // Side, Rectangle, Circle, Difference, and Union.
#if 1
  bc->determineIteratorLimits(face,this);
#else
  bc->oldDetermineIteratorLimits(face,this);
#endif
  array_bcs[face] = bc->clone();
}  
 
const BCDataArray* Patch::getBCDataArray(Patch::FaceType face) const
{
  map<Patch::FaceType,BCDataArray*>::const_iterator itr = array_bcs.find(face);
  if (itr != array_bcs.end())
    return itr->second;
  else
    return 0;
}

const BoundCondBase*
Patch::getArrayBCValues(Patch::FaceType face,
                        int mat_id,
                        const string& type,
			vector<IntVector>*& bound_ptr, 
			vector<IntVector>*& nbound_ptr,
			int child) const
{
  map<Patch::FaceType,BCDataArray* >::const_iterator itr=array_bcs.find(face); 
  if (itr != array_bcs.end()) {
    const BoundCondBase* bc = itr->second->getBoundCondData(mat_id,type,child);
    itr->second->getBoundaryIterator( mat_id,bound_ptr,  child);
    itr->second->getNBoundaryIterator(mat_id,nbound_ptr, child);
    return bc;
  } else
    return 0;
}

bool 
Patch::haveBC(FaceType face,int mat_id,const string& bc_type,
              const string& bc_variable) const
{
  map<Patch::FaceType,BCDataArray* >::const_iterator itr=array_bcs.find(face); 

  if (itr != array_bcs.end()) {
#if 0
    cout << "Inside haveBC" << endl;
    ubc->print();
#endif
    BCDataArray::bcDataArrayType::const_iterator v_itr;
    vector<BCGeomBase*>::const_iterator it;
    
    v_itr = itr->second->d_BCDataArray.find(mat_id);
    if (v_itr != itr->second->d_BCDataArray.end()) { 
      for (it = v_itr->second.begin(); it != v_itr->second.end(); ++it) {
      BCData bc;
      (*it)->getBCData(bc);
      bool found_variable = bc.find(bc_type,bc_variable);
      if (found_variable)
	return true;
      }
    } 
    // Check the mat_it = "all" case
    v_itr = itr->second->d_BCDataArray.find(-1);
    if (v_itr != itr->second->d_BCDataArray.end()) {
      for (it = v_itr->second.begin(); it != v_itr->second.end(); ++it) {
	BCData bc;
	(*it)->getBCData(bc);
	bool found_variable = bc.find(bc_type,bc_variable);
	if (found_variable)
	  return true;
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
  IntVector ll=getInteriorCellLowIndex();
  IntVector hh=getInteriorCellHighIndex();
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceExtraNodes", __FILE__, __LINE__));
  }
}

void
Patch::getFaceNodes(FaceType face, int offset,IntVector& l, IntVector& h) const
{
  // Change from getNodeLowIndex to getInteriorNodeLowIndex.  Need to do this
  // when we have extra cells.
  IntVector lorig=l=getInteriorNodeLowIndex();
  IntVector horig=h=getInteriorNodeHighIndex();
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
   IntVector lorig=l=getCellLowIndex();
   IntVector horig=h=getCellHighIndex();
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

  Box box(getBox());
  sprintf( str, "[ [%2.2f, %2.2f, %2.2f] [%2.2f, %2.2f, %2.2f] ]",
	   box.lower().x(), box.lower().y(), box.lower().z(),
	   box.upper().x(), box.upper().y(), box.upper().z() );

  return string( str );
}

/*
void Patch::setExtraIndices(const IntVector& l, const IntVector& h)
{
  d_lowIndex = l;
  d_highIndex = h;
}
*/

// This function will return all cells that are intersected by the
// box.  This is based on the fact that boundaries of cells are closed
// on the bottom and open on the top.
CellIterator
Patch::getCellIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low(RoundDown(l.x()), RoundDown(l.y()), RoundDown(l.z()));
   // high is the inclusive upper bound on the index.  In order for
   // the iterator to work properly we need in increment all the
   // indices by 1.
   IntVector high(RoundDown(u.x())+1, RoundDown(u.y())+1, RoundDown(u.z())+1);
   low = Max(low, getCellLowIndex());
   high = Min(high, getCellHighIndex());
   return CellIterator(low, high);
}

// This function works on the assumption that we want all the cells
// whose centers lie on or within the box.
CellIterator
Patch::getCellCenterIterator(const Box& b) const
{
#if 1
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   // If we subtract 0.5 from the bounding box locations we can treat
   // the code just like we treat nodes.
   l -= Vector(0.5, 0.5, 0.5);
   u -= Vector(0.5, 0.5, 0.5);
   // This will return an empty iterator when the box is degerate.
   IntVector low(RoundUp(l.x()), RoundUp(l.y()), RoundUp(l.z()));
   IntVector high(RoundDown(u.x()) + 1, RoundDown(u.y()) + 1,
		  RoundDown(u.z()) + 1);
#else
  // This is the old code, which doesn't really follow the
  // specifiaction, but works the way some people have gotten used to
  // it working.
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low(RoundDown(l.x()), RoundDown(l.y()), RoundDown(l.z()));
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
#endif
   low = Max(low, getCellLowIndex());
   high = Min(high, getCellHighIndex());
   return CellIterator(low, high);
}

CellIterator
Patch::getExtraCellIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low(RoundDown(l.x()), RoundDown(l.y()), RoundDown(l.z()));
   IntVector high(RoundUp(u.x()),  RoundUp(u.y()),   RoundUp(u.z()));
   low = Min(low, getCellLowIndex());
   high = Max(high, getCellHighIndex());
   return CellIterator(low, high);
}

CellIterator
Patch::getCellIterator(const IntVector gc) const
{
  //   return CellIterator(getCellLowIndex(), getCellHighIndex());
   return CellIterator(d_inLowIndex-gc, d_inHighIndex+gc);
}


CellIterator
Patch::getExtraCellIterator(const IntVector gc) const
{
  return CellIterator(getCellLowIndex()-gc, getCellHighIndex()+gc);
}

//______________________________________________________________________
//       I C E  /  M P M I C E   I T E R A T O R S
//
//  For SFCXFace Variables
//  Iterates over all interior facing cell faces
#if 1
CellIterator
Patch::getSFCXIterator(const int offset) const
{
  IntVector low,hi; 
  low = d_inLowIndex;
  low+=IntVector(getBCType(Patch::xminus)==Neighbor?0:offset,
		   getBCType(Patch::yminus)==Neighbor?0:0,
		   getBCType(Patch::zminus)==Neighbor?0:0);
  hi  = d_inHighIndex + IntVector(1,0,0);
  hi -=IntVector(getBCType(Patch::xplus) ==Neighbor?1:offset,
		   getBCType(Patch::yplus) ==Neighbor?0:0,
		   getBCType(Patch::zplus) ==Neighbor?0:0);
  return CellIterator(low, hi);
}
//__________________________________
//  Iterates over all interior facing cell faces
CellIterator
Patch::getSFCYIterator(const int offset) const
{
  IntVector low,hi; 
  low = d_inLowIndex;
  low+=IntVector(getBCType(Patch::xminus)==Neighbor?0:0,
		   getBCType(Patch::yminus)==Neighbor?0:offset,
		   getBCType(Patch::zminus)==Neighbor?0:0);
  hi  = d_inHighIndex + IntVector(0,1,0);
  hi -=IntVector(getBCType(Patch::xplus) ==Neighbor?0:0,
		   getBCType(Patch::yplus) ==Neighbor?1:offset,
		   getBCType(Patch::zplus) ==Neighbor?0:0);
  return CellIterator(low, hi);
}
//__________________________________
//  Iterates over all interior facing cell faces
CellIterator
Patch::getSFCZIterator(const int offset) const
{
  IntVector low,hi; 
  low = d_inLowIndex;
  low+=IntVector(getBCType(Patch::xminus)==Neighbor?0:0,
		   getBCType(Patch::yminus)==Neighbor?0:0,
		   getBCType(Patch::zminus)==Neighbor?0:offset);
  hi  = d_inHighIndex +   IntVector(0,0,1);
  hi -=IntVector(getBCType(Patch::xplus) ==Neighbor?0:0,
		   getBCType(Patch::yplus) ==Neighbor?0:0,
		   getBCType(Patch::zplus) ==Neighbor?1:offset);
  return CellIterator(low, hi);
}
#endif
//__________________________________
// Selects which iterator to use
//  based on direction
CellIterator
Patch::getSFCIterator(const int dir, const int offset) const
{
  if (dir == 0) {
    return getSFCXIterator(offset);
  } else if (dir == 1) {
    return getSFCYIterator(offset);
  } else if (dir == 2) {
    return getSFCZIterator(offset);
  } else {
    SCI_THROW(InternalError("Patch::getSFCIterator: dir must be 0, 1, or 2", __FILE__, __LINE__));
  }
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
Patch::getFaceIterator__New(const FaceType& face, const FaceIteratorType& domain) const
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
      lowPt =  getCellLowIndex__New();
      highPt = getCellHighIndex__New();

      //select the face
      switch(plusface)
      {
        case true:
          //restrict dimension to the face
          lowPt[dim]=highPt[dim];
          //extend dimension by extra cells
          highPt[dim]=getExtraCellHighIndex__New()[dim];
          break;
        case false:
          //restrict dimension to face
          highPt[dim]=lowPt[dim];
          //extend dimension by extra cells
          lowPt[dim]=getExtraCellLowIndex__New()[dim];
          break;
      }
      break;
      //start with the loose fitting patch and contract the indices to exclude the unwanted regions
    case ExtraPlusEdgeCells:
      //grab patch region with extra cells
      lowPt =  getExtraCellLowIndex__New();
      highPt = getExtraCellHighIndex__New();
     
      //select the face
      switch(plusface)
      {
        case true:
          //move low point to plus face
          lowPt[dim]=getCellHighIndex__New()[dim];
          break;
        case false:
          //move high point to minus face
          highPt[dim]=getCellLowIndex__New()[dim];
          break;
      }
      break;
    case FaceNodes:
      //grab patch region without extra cells
      lowPt =  getNodeLowIndex__New();
      highPt = getNodeHighIndex__New();

      //select the face
      switch(plusface)
      {
        case true:
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //extend low point by 1 cell
          lowPt[dim]=lowPt[dim]-1;
          break;
        case false:
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //extend high point by 1 cell
          highPt[dim]=highPt[dim]+1;
          break;
      }
      break;
    case SFCVars:
     
      //grab patch region without extra cells
      switch(dim)
      {
        case 0:
          lowPt =  getSFCXLowIndex__New();
          highPt = getSFCXHighIndex__New();
          break;

        case 1:
          lowPt =  getSFCYLowIndex__New();
          highPt = getSFCYHighIndex__New();
          break;

        case 2:
          lowPt =  getSFCZLowIndex__New();
          highPt = getSFCZHighIndex__New();
          break;
      }

      //select the face
      switch(plusface)
      {
        case true:
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //extend low point by 1 cell
          lowPt[dim]=lowPt[dim]-1;
          break;
        case false:
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //extend high point by 1 cell
          highPt[dim]=highPt[dim]+1;
          break;
      }
      break;
    case InteriorFaceCells:
      lowPt =  getCellLowIndex__New();
      highPt = getCellHighIndex__New();
      
      //select the face
      switch(plusface)
      {
        case true:
          //restrict index to face
          lowPt[dim]=highPt[dim];
          //contract dimension by 1
          lowPt[dim]--;
          break;
        case false:
          //restrict index to face
          highPt[dim]=lowPt[dim];
          //contract dimension by 1
          highPt[dim]++;
          break;
      }
      break;
    default:
      throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
  }
  return CellIterator(lowPt, highPt);
}
//__________________________________
//  Iterate over the GhostCells on a particular face
// domain:  
//  plusEdgeCells:          Includes the edge and corner cells.
//  NC_vars/FC_vars:        Hit all nodes/faces on the border of the extra cells        
//  alongInteriorFaceCells: Hit interior face cells                                                            
CellIterator    
Patch::getFaceCellIterator(const FaceType& face, const string& domain) const
{ 
  IntVector lowPt  = d_inLowIndex;   // interior low
  IntVector highPt = d_inHighIndex;  // interior high
  int offset = 0;
  
  // Controls if the iterator hits the extra cells or interior cells.
  // default is to hit the extra cells. 
  int shiftInward = 0;
  int shiftOutward = 1;     
 
  //__________________________________
  // different options
  if(domain == "plusEdgeCells"){ 
    lowPt   = d_lowIndex;
    highPt  = d_highIndex;
  }
  if(domain == "NC_vars"){  
    lowPt   = d_inLowIndex;
    highPt  = d_highIndex;
    offset  = 1;
  }
  if(domain == "FC_vars"){  
    lowPt   = d_lowIndex;
    highPt  = d_highIndex;
    offset  = 1;
  }
  if(domain == "alongInteriorFaceCells"){
    lowPt   = d_inLowIndex;
    highPt  = d_inHighIndex;
    offset  = 0;
    shiftInward = 1;
    shiftOutward = 0;
  }

  if (face == Patch::xplus) {           //  X P L U S
    lowPt.x(d_inHighIndex.x() - shiftInward);
    highPt.x(d_inHighIndex.x()+ shiftOutward);
  }
  if(face == Patch::xminus){            //  X M I N U S
    lowPt.x(d_inLowIndex.x()  - shiftOutward + offset);
    highPt.x(d_inLowIndex.x() + shiftInward  + offset);
  }
  if(face == Patch::yplus) {            //  Y P L U S
    lowPt.y(d_inHighIndex.y()  - shiftInward);
    highPt.y(d_inHighIndex.y() + shiftOutward);
  }
  if(face == Patch::yminus) {           //  Y M I N U S
    lowPt.y(d_inLowIndex.y()   - shiftOutward + offset);
    highPt.y(d_inLowIndex.y()  + shiftInward  + offset);
  }
  if (face == Patch::zplus) {           //  Z P L U S
    lowPt.z(d_inHighIndex.z()  - shiftInward);
    highPt.z(d_inHighIndex.z() + shiftOutward);
  }
  if (face == Patch::zminus) {          //  Z M I N U S
    lowPt.z(d_inLowIndex.z()   - shiftOutward + offset);
    highPt.z(d_inLowIndex.z()  + shiftInward  + offset);
  } 
  return CellIterator(lowPt, highPt);
}

/**
 * Returns an iterator to the edge of two intersecting faces.
 * if minusCornerCells is true the edge will exclude corner cells.
 */
CellIterator    
Patch::getEdgeCellIterator__New(const FaceType& face0, 
                           const FaceType& face1,bool minusCornerCells) const
{
  FaceType face[2]={face0,face1};

  //will contain extra cells
  vector<IntVector>loPt(2), hiPt(2); 
  loPt[0] = loPt[1] = getExtraCellLowIndex__New();
  hiPt[0] = hiPt[1] = getExtraCellHighIndex__New();

  int dim[2]={getFaceDimension(face0),getFaceDimension(face1)};

  //return an empty iterator if trying to intersect the same dimension
  if(dim[0]==dim[1])
    return CellIterator(IntVector(0,0,0),IntVector(0,0,0));

  for (int f = 0; f < 2 ; f++ ) {
   
    switch(face[f])
    {
        case xminus: case yminus: case zminus:
          //restrict to face
          hiPt[f][dim[f]]=getCellLowIndex__New()[dim[f]];
          break;
        case xplus: case yplus: case zplus:
          //restrict to face
          loPt[f][dim[f]]=getCellHighIndex__New()[dim[f]];
          break;
        default:
          throw InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
          break;

    }

    //remove the corner cells by pruning the other dimension's extra cells
    if(minusCornerCells)
    {
      //compute the dimension of the face not being used
      int otherdim=3-dim[0]-dim[1];
  
      loPt[f][otherdim]=getCellLowIndex__New()[otherdim];
      hiPt[f][otherdim]=getCellHighIndex__New()[otherdim];
    }
  }

  // compute the edge low and high pt from the intersection
  IntVector LowPt  = Max(loPt[0], loPt[1]);
  IntVector HighPt = Min(hiPt[0], hiPt[1]);
  
  return CellIterator(LowPt, HighPt);
  

}
//__________________________________
//  Iterate over an edge at the intersection of face0 and face1
// if domain == minusCornerCells this subtracts off the corner cells.
CellIterator    
Patch::getEdgeCellIterator(const FaceType& face0, 
                           const FaceType& face1,const string& domain) const
{ 
  vector<IntVector>loPt(2);   
  vector<IntVector>hiPt(2); 
  loPt[0] = d_lowIndex;
  loPt[1] = d_lowIndex;
  hiPt[0] = d_highIndex;
  hiPt[1] = d_highIndex;
  
  IntVector dir0 = faceDirection(face0);
  IntVector dir1 = faceDirection(face1);
  IntVector test = dir0 + dir1; // add the normal components, i.e.,
                                // xminus(-1) + xplus(1) = 0
  
  if (face0 == face1 || test == IntVector(0,0,0)) {  //  no edge here
    return CellIterator(loPt[0], loPt[0]);
  }
  
  for (int f = 0; f < 2 ; f++ ) {
    FaceType face;
    if (f == 0 ) {
     face = face0;
    }else{
     face = face1;
    }
    
    if(face == Patch::xplus) {           //  X P L U S
      loPt[f].x(d_inHighIndex.x());
      hiPt[f].x(d_inHighIndex.x()+1);
    }
    if(face == Patch::xminus){           //  X M I N U S
      loPt[f].x(d_inLowIndex.x()-1);
      hiPt[f].x(d_inLowIndex.x());
    }
    if(face == Patch::yplus) {           //  Y P L U S
      loPt[f].y(d_inHighIndex.y());
      hiPt[f].y(d_inHighIndex.y()+1);
    }
    if(face == Patch::yminus) {          //  Y M I N U S
      loPt[f].y(d_inLowIndex.y()-1);
      hiPt[f].y(d_inLowIndex.y());
    }
    if(face == Patch::zplus) {           //  Z P L U S
      loPt[f].z(d_inHighIndex.z() );
      hiPt[f].z(d_inHighIndex.z()+1);
    }
    if(face == Patch::zminus) {          //  Z M I N U S
      loPt[f].z(d_inLowIndex.z()-1);
      hiPt[f].z(d_inLowIndex.z());
    } 
  }
  // compute the edge low and high pt from the intersection
  IntVector LowPt  = Max(loPt[0], loPt[1]);
  IntVector HighPt = Min(hiPt[0], hiPt[1]);
  
  if(domain == "minusCornerCells"){
    IntVector offset = IntVector(1,1,1) - Abs(dir0) - Abs(dir1);
    
    vector<IntVector> corner; 
    getCornerCells(corner,face0);
    vector<IntVector>::const_iterator itr;

    for(itr = corner.begin(); itr != corner.end(); ++ itr ) {
      IntVector corner = *itr;
      if (corner == LowPt) {
        LowPt += offset;
      }
      if (corner == (HighPt - IntVector(1,1,1)) ) {
        HighPt -= offset;
      }
    }
  }
  return CellIterator(LowPt, HighPt);
}


//__________________________________
//   Expands the cell iterator (hi_lo)
//  into the (nCells) ghost cells for each patch
CellIterator    
Patch::addGhostCell_Iter(CellIterator hi_lo, const int nCells) const                                        
{
  IntVector low,hi; 
  low = hi_lo.begin();
  hi  = hi_lo.end();
  IntVector ll(low);
  IntVector hh(hi);
  ll -= IntVector(getBCType(Patch::xminus) == Neighbor?nCells:0,
		    getBCType(Patch::yminus) == Neighbor?nCells:0,  
		    getBCType(Patch::zminus) == Neighbor?nCells:0); 

  hh += IntVector(getBCType(Patch::xplus) == Neighbor?nCells:0,
		    getBCType(Patch::yplus) == Neighbor?nCells:0,
		    getBCType(Patch::zplus) == Neighbor?nCells:0);
  
   return  CellIterator(ll,hh);
} 


Box Patch::getGhostBox(const IntVector& lowOffset,
		       const IntVector& highOffset) const
{
   return Box(d_level->getNodePosition(d_lowIndex+lowOffset),
	      d_level->getNodePosition(d_highIndex+highOffset));
}

NodeIterator Patch::getNodeIterator() const
{
  IntVector low = d_inLowIndex;
  IntVector hi = d_inHighIndex +
    IntVector(getBCType(xplus) == Neighbor?0:1,
              getBCType(yplus) == Neighbor?0:1,
              getBCType(zplus) == Neighbor?0:1);
  //   return NodeIterator(getNodeLowIndex(), getNodeHighIndex());

  return NodeIterator(low, hi);
}

// This will return an iterator which will include all the nodes
// contained by the bounding box.  If a dimension of the widget is
// degenerate (has a thickness of 0) the nearest node in that
// dimension is used.
NodeIterator
Patch::getNodeIterator(const Box& b) const
{
  // Determine if we are dealing with a 2D box.
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
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
   low = Max(low, getNodeLowIndex());
   high = Min(high, getNodeHighIndex());
   return NodeIterator(low, high);
}

NodeIterator Patch::getNodeIterator(const string& interp_type) const
{
  if(interp_type!="gimp" && interp_type!="3rdorderBS"){
   return getNodeIterator();
  }
  else{
    IntVector low = d_inLowIndex -
    IntVector(getBCType(xminus) == Neighbor?0:1,
              getBCType(yminus) == Neighbor?0:1,
              getBCType(zminus) == Neighbor?0:1);

    IntVector hi = d_inHighIndex +
    IntVector(getBCType(xplus) == Neighbor?0:2,
              getBCType(yplus) == Neighbor?0:2,
              getBCType(zplus) == Neighbor?0:2);

    return NodeIterator(low, hi);
  }
}

IntVector Patch::getInteriorNodeLowIndex() const {
    return d_inLowIndex;
}

IntVector Patch::getInteriorNodeHighIndex() const {
   IntVector hi = d_inHighIndex;
   hi +=IntVector(getBCType(xplus) == Neighbor?0:1,
                  getBCType(yplus) == Neighbor?0:1,
                  getBCType(zplus) == Neighbor?0:1);
   return hi;
 }

IntVector Patch::getSFCXHighIndex() const
{
   IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0));
   return h;
}

IntVector Patch::getSFCYHighIndex() const
{
   IntVector h(d_highIndex+
	       IntVector(0, getBCType(yplus) == Neighbor?0:1, 0));
   return h;
}

IntVector Patch::getSFCZHighIndex() const
{
   IntVector h(d_highIndex+
	       IntVector(0, 0, getBCType(zplus) == Neighbor?0:1));
   return h;
}
// if next to a boundary then lowIndex = 2+celllowindex in the flow dir
IntVector Patch::getSFCXFORTLowIndex() const
{
  IntVector h(d_lowIndex+
	      IntVector(getBCType(xminus) == Neighbor?0:2, 
			getBCType(yminus) == Neighbor?0:1,
			getBCType(zminus) == Neighbor?0:1));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCXFORTHighIndex() const
{
   IntVector h(d_highIndex - IntVector(1,1,1) - 
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1));
   return h;
}

// if next to a boundary then lowIndex = 2+celllowindex
IntVector Patch::getSFCYFORTLowIndex() const
{
  IntVector h(d_lowIndex+
	      IntVector(getBCType(xminus) == Neighbor?0:1, 
			getBCType(yminus) == Neighbor?0:2,
			getBCType(zminus) == Neighbor?0:1));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCYFORTHighIndex() const
{
   IntVector h(d_highIndex - IntVector(1,1,1) - 
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1));
   return h;
}

// if next to a boundary then lowIndex = 2+celllowindex
IntVector Patch::getSFCZFORTLowIndex() const
{
  IntVector h(d_lowIndex+
	      IntVector(getBCType(xminus) == Neighbor?0:1, 
			getBCType(yminus) == Neighbor?0:1,
			getBCType(zminus) == Neighbor?0:2));
  return h;
}
// if next to a boundary then highindex = cellhighindex - 1 - 1(coz of fortran)
IntVector Patch::getSFCZFORTHighIndex() const
{
   IntVector h(d_highIndex - IntVector(1,1,1) - 
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1));
   return h;
}
  
IntVector Patch::getCellFORTLowIndex() const
{
 IntVector h(d_lowIndex+
	      IntVector(getBCType(xminus) == Neighbor?0:1, 
			getBCType(yminus) == Neighbor?0:1,
			getBCType(zminus) == Neighbor?0:1));
  return h;
  
}
IntVector Patch::getCellFORTHighIndex() const
{
   IntVector h(d_highIndex - IntVector(1,1,1) - 
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1));
   return h;

}

IntVector Patch::getGhostCellLowIndex(int numGC) const
{
  return d_lowIndex - IntVector(getBCType(xminus) == Neighbor?numGC:0,
				getBCType(yminus) == Neighbor?numGC:0,
				getBCType(zminus) == Neighbor?numGC:0);

}

IntVector Patch::getGhostCellHighIndex(int numGC) const
{
  return d_highIndex + IntVector(getBCType(xplus) == Neighbor?numGC:0,
				 getBCType(yplus) == Neighbor?numGC:0,
				 getBCType(zplus) == Neighbor?numGC:0);
}

void Patch::cullIntersection(VariableBasis basis, IntVector bl, const Patch* neighbor,
                             IntVector& region_low, IntVector& region_high) const
{
  // on certain AMR grid configurations, with extra cells, patches can overlap
  // such that the extra cell of one patch overlaps a normal cell of another
  // in such conditions, we shall exclude that extra cell from MPI communication
  // Also disclude overlapping extra cells just to be safe

  // follow this heuristic - if one dimension of neighbor overlaps "this" with 2 cells
  //   then prune it back to its interior cells
  //   if two or more, throw it out entirely.  If there are 2+ different, the patches share only a line or point
  //   and the patch's boundary conditions are basically as though there were no patch there

  // use the cell-based interior to compare patch positions, but use the basis-specific one when culling the intersection
  IntVector p_int_low(getInteriorLowIndex(Patch::CellBased)), p_int_high(getInteriorHighIndex(Patch::CellBased));
  IntVector n_int_low(neighbor->getInteriorLowIndex(Patch::CellBased)), n_int_high(neighbor->getInteriorHighIndex(Patch::CellBased));

  // actual patch intersection
  IntVector diff = Abs(Max(getLowIndex(Patch::CellBased, bl), neighbor->getLowIndex(Patch::CellBased, bl)) -
                       Min(getHighIndex(Patch::CellBased, bl), neighbor->getHighIndex(Patch::CellBased, bl)));

  // go through each dimension, and determine where the neighor patch is relative to this
  // if it is above or below, clamp it to the interior of the neighbor patch
  // based on the current grid constraints, it is reasonable to assume that the patches
  // line up at least in corners.
  int bad_diffs = 0;
  for (int dim = 0; dim < 3; dim++) {
    if (diff[dim] == 2) // if it's two, then it must be overlapping extra cells (min patch size is 3, even in 1/2D)
      bad_diffs++;
    // depending on the region, cull away the portion of the region that in 'this'
    if (n_int_high[dim] == p_int_low[dim]) {
      region_high[dim] = Min(region_high[dim], neighbor->getInteriorHighIndex(basis)[dim]);
    }
    else if (n_int_low[dim] == p_int_high[dim]) {
      region_low[dim] = Max(region_low[dim], neighbor->getInteriorLowIndex(basis)[dim]);
    }
  }
  
  // prune it back if heuristic met or if already empty
  IntVector region_diff = region_high - region_low;
  if (bad_diffs >= 2 || region_diff.x() * region_diff.y() * region_diff.z() == 0)
    region_low = region_high;  // caller will check for this case

}

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
  d_level->selectPatches(low, high, neighbors);
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
  
  IntVector origLowIndex = getLowIndex(basis, IntVector(0,0,0));
  IntVector origHighIndex = getHighIndex(basis, IntVector(0,0,0));
  low = origLowIndex - lowOffset;
  high = origHighIndex + highOffset;

  for (int i = 0; i < 3; i++) {
    FaceType faceType = (FaceType)(2 * i); // x, y, or z minus
    if (getBCType(faceType) != Neighbor) {
      // no neighbor -- use original low index for that side
      low[i] = origLowIndex[i]-boundaryLayer[i];
    }
    
    faceType = (FaceType)(faceType + 1); // x, y, or z plus
    if (getBCType(faceType) != Neighbor) {
      // no neighbor -- use original high index for that side
      high[i] = origHighIndex[i]+boundaryLayer[i];
    }
  }
}

void Patch::getOtherLevelPatches(int levelOffset,
				 Patch::selectType& selected_patches,
                                 int numGhostCells /*=0*/) const
{
  ASSERT(levelOffset == 1 || levelOffset == -1);

  // include in the final low/high
  IntVector gc(numGhostCells, numGhostCells, numGhostCells);

  const LevelP& otherLevel = d_level->getRelativeLevel(levelOffset);
  IntVector low = 
    otherLevel->getCellIndex(d_level->getCellPosition(getLowIndex()));
  IntVector high =
    otherLevel->getCellIndex(d_level->getCellPosition(getHighIndex()));

  if (levelOffset < 0) {
    // we don't grab enough in the high direction if the fine extra cell
    // is on the other side of a coarse boundary

    // refinement ratio between the two levels
    IntVector crr = otherLevel->getRelativeLevel(1)->getRefinementRatio();
    IntVector highIndex = getHighIndex();
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

  //cout << "  Patch:Golp: " << low-gc << " " << high+gc << endl;
  Level::selectType patches;
  otherLevel->selectPatches(low-gc, high+gc, patches); 
  
  // based on the expanded range above to search for extra cells, we might
  // have grabbed more patches than we wanted, so refine them here
  
  for (int i = 0; i < patches.size(); i++) {
    if (levelOffset < 0 || getBox().overlaps(patches[i]->getBox())) {
      selected_patches.push_back(patches[i]);
    }
  }
}

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
      SCI_THROW(InternalError("Unknown variable type in Patch::getVariableExtents (from TypeDescription::Type)",
                              __FILE__, __LINE__));
    else
      return CellBased; // doesn't matter
  }
}

Box Patch::getBox() const {
  return d_level->getBox(d_lowIndex, d_highIndex);
}

Box Patch::getInteriorBox() const {
  return d_level->getBox(d_inLowIndex, d_inHighIndex);
}

IntVector Patch::neighborsLow() const
{
  return IntVector(getBCType(xminus) == Neighbor? 0:1,
		   getBCType(yminus) == Neighbor? 0:1,
		   getBCType(zminus) == Neighbor? 0:1);
}

IntVector Patch::neighborsHigh() const
{
  return IntVector(getBCType(xplus) == Neighbor? 0:1,
		   getBCType(yplus) == Neighbor? 0:1,
		   getBCType(zplus) == Neighbor? 0:1);
}

IntVector Patch::getLowIndex(VariableBasis basis,
			     const IntVector& boundaryLayer) const
{
  switch (basis) {
  case CellBased:
    return getCellLowIndex()-neighborsLow()*boundaryLayer;
  case NodeBased:
    return getNodeLowIndex()-neighborsLow()*boundaryLayer;
  case XFaceBased:
    return getSFCXLowIndex()-neighborsLow()*boundaryLayer;
  case YFaceBased:
    return getSFCYLowIndex()-neighborsLow()*boundaryLayer;
  case ZFaceBased:
    return getSFCZLowIndex()-neighborsLow()*boundaryLayer;
  case AllFaceBased:
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  }
}

IntVector Patch::getHighIndex(VariableBasis basis,
			      const IntVector& boundaryLayer) const
{
  switch (basis) {
  case CellBased:
    return getCellHighIndex()+neighborsHigh()*boundaryLayer;
  case NodeBased:
    return getNodeHighIndex()+neighborsHigh()*boundaryLayer;
  case XFaceBased:
    return getSFCXHighIndex()+neighborsHigh()*boundaryLayer;
  case YFaceBased:
    return getSFCYHighIndex()+neighborsHigh()*boundaryLayer;
  case ZFaceBased:
    return getSFCZHighIndex()+neighborsHigh()*boundaryLayer;
  case AllFaceBased:
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)",
                            __FILE__, __LINE__));
  }
}

IntVector Patch::getInteriorLowIndex(VariableBasis basis) const
{
  return d_inLowIndex;
}

IntVector Patch::getInteriorHighIndex(VariableBasis basis) const
{
  switch (basis) {
  case CellBased:
    return d_inHighIndex;
  case NodeBased:
    return getInteriorNodeHighIndex();
  case XFaceBased:
    return d_inHighIndex+IntVector(getBCType(xplus) == Neighbor? 0:1,0,0);
  case YFaceBased:
    return d_inHighIndex+IntVector(0,getBCType(yplus) == Neighbor? 0:1,0);
  case ZFaceBased:
    return d_inHighIndex+IntVector(0,0,getBCType(zplus) == Neighbor? 0:1);
  case AllFaceBased:
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getInteriorHighIndex(basis)",
                            __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getInteriorHighIndex(basis)",
                            __FILE__, __LINE__));
  }
}

IntVector Patch::getInteriorLowIndexWithBoundary(VariableBasis basis) const
{
  IntVector inlow = getInteriorLowIndex(basis);
  IntVector low = getLowIndex(basis, IntVector(0,0,0));
  if (getBCType(xminus) == None) inlow[0] = low[0];
  if (getBCType(yminus) == None) inlow[1] = low[1];
  if (getBCType(zminus) == None) inlow[2] = low[2];
  return inlow;
}

IntVector Patch::getInteriorHighIndexWithBoundary(VariableBasis basis) const
{
  IntVector inhigh = getInteriorHighIndex(basis);
  IntVector high = getHighIndex(basis, IntVector(0,0,0));
  if (getBCType(xplus) == None) inhigh[0] = high[0];
  if (getBCType(yplus) == None) inhigh[1] = high[1];
  if (getBCType(zplus) == None) inhigh[2] = high[2];
  return inhigh;
}

void Patch::finalizePatch()
{
  TAU_PROFILE("Patch::finalizePatch()", " ", TAU_USER);
  
  //set the grid index
  d_patchState.gridIndex=d_newGridIndex;
  //set the level index
  
#if SCI_ASSERTION_LEVEL>0
  ASSERT(getLow()==getExtraCellLowIndex__New());
  ASSERT(getHigh()==getExtraCellHighIndex__New());
  ASSERT(getLowIndex()==getExtraCellLowIndex__New());
  ASSERT(getHighIndex()==getExtraCellHighIndex__New());
  ASSERT(getNodeLowIndex()==getExtraNodeLowIndex__New());
  ASSERT(getNodeHighIndex()==getExtraNodeHighIndex__New());
  ASSERT(getInteriorNodeLowIndex()==getNodeLowIndex__New());
  ASSERT(getInteriorNodeHighIndex()==getNodeHighIndex__New());
  ASSERT(getCellLowIndex()==getExtraCellLowIndex__New());
  ASSERT(getCellHighIndex()==getExtraCellHighIndex__New());
  ASSERT(getInteriorCellLowIndex()==getCellLowIndex__New());
  ASSERT(getInteriorCellHighIndex()==getCellHighIndex__New());
  ASSERT(getGhostCellLowIndex(1)==getExtraCellLowIndex__New(1));
  ASSERT(getGhostCellHighIndex(1)==getExtraCellHighIndex__New(1));
  ASSERT(getCellIterator().begin()==getCellIterator__New().begin());
  ASSERT(getCellIterator().end()==getCellIterator__New().end());
  ASSERT(getExtraCellIterator().begin()==getExtraCellIterator__New().begin());
  ASSERT(getExtraCellIterator().end()==getExtraCellIterator__New().end());
  ASSERT(getNodeIterator().begin()==getNodeIterator__New().begin());
  ASSERT(getNodeIterator().end()==getNodeIterator__New().end());
  ASSERT(getSFCXLowIndex()==getExtraSFCXLowIndex__New());
  ASSERT(getSFCXHighIndex()==getExtraSFCXHighIndex__New());
  ASSERT(getSFCYLowIndex()==getExtraSFCYLowIndex__New());
  ASSERT(getSFCYHighIndex()==getExtraSFCYHighIndex__New());
  ASSERT(getSFCZLowIndex()==getExtraSFCZLowIndex__New());
  ASSERT(getSFCZHighIndex()==getExtraSFCZHighIndex__New());
  ASSERT(getSFCXIterator().begin()==getSFCXIterator__New().begin());
  ASSERT(getSFCXIterator().end()==getSFCXIterator__New().end());
  ASSERT(getSFCYIterator().begin()==getSFCYIterator__New().begin());
  ASSERT(getSFCYIterator().end()==getSFCYIterator__New().end());
  ASSERT(getSFCZIterator().begin()==getSFCZIterator__New().begin());
  ASSERT(getSFCZIterator().end()==getSFCZIterator__New().end());
  
  ASSERT(d_extraCells!=IntVector(1,1,1) || getCellFORTLowIndex()==getFortranCellLowIndex__New());
  ASSERT(d_extraCells!=IntVector(1,1,1) ||getCellFORTHighIndex()==getFortranCellHighIndex__New());

  //ASSERT(getSFCXFORTLowIndex()==getFortranZFC_ExtraCellLowIndex__New());
  //ASSERT(getSFCXFORTHighIndex()==getFortranZFC_ExtraCellHighIndex__New());
  //ASSERT(getSFCYFORTLowIndex()==getFortranZFC_ExtraCellLowIndex__New());
  //ASSERT(getSFCYFORTHighIndex()==getFortranZFC_ExtraCellHighIndex__New());
  //ASSERT(getSFCZFORTLowIndex()==getFortranZFC_ExtraCellLowIndex__New());
  //ASSERT(getSFCZFORTHighIndex()==getFortranZFC_ExtraCellHighIndex__New());

  /*
  FaceType face=xplus;
  string domain1="NC_vars";
  FaceIteratorType domain2=FaceNodes;
  if(d_extraCells!=IntVector(1,1,1) || getBCType(face)!=Neighbor && !(getFaceCellIterator(face,domain1)==getFaceIterator__New(face,domain2)))
  {
      cout << "old:" << getFaceCellIterator(face,domain1).begin() << " -> " << getFaceCellIterator(face,domain1).end() << endl;
      cout << "new:" << getFaceIterator__New(face,domain2).begin() << " -> " << getFaceIterator__New(face,domain2).end() << endl;
      for(int i=0;i<6;i++)
      {
        cout << "face " << i << ":" << getBCType(static_cast<FaceType>(i)) << endl;
      }
  }
  */


  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getFaceCellIterator(xplus,"minusEdgeCells")==getFaceIterator__New(xplus,ExtraMinusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getFaceCellIterator(xminus,"minusEdgeCells")==getFaceIterator__New(xminus,ExtraMinusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yplus)==Neighbor || getFaceCellIterator(yplus,"minusEdgeCells")==getFaceIterator__New(yplus,ExtraMinusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yminus)==Neighbor || getFaceCellIterator(yminus,"minusEdgeCells")==getFaceIterator__New(yminus,ExtraMinusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zplus)==Neighbor || getFaceCellIterator(zplus,"minusEdgeCells")==getFaceIterator__New(zplus,ExtraMinusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zminus)==Neighbor || getFaceCellIterator(zminus,"minusEdgeCells")==getFaceIterator__New(zminus,ExtraMinusEdgeCells));
  
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getFaceCellIterator(xplus,"plusEdgeCells")==getFaceIterator__New(xplus,ExtraPlusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getFaceCellIterator(xminus,"plusEdgeCells")==getFaceIterator__New(xminus,ExtraPlusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yplus)==Neighbor || getFaceCellIterator(yplus,"plusEdgeCells")==getFaceIterator__New(yplus,ExtraPlusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yminus)==Neighbor || getFaceCellIterator(yminus,"plusEdgeCells")==getFaceIterator__New(yminus,ExtraPlusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zplus)==Neighbor || getFaceCellIterator(zplus,"plusEdgeCells")==getFaceIterator__New(zplus,ExtraPlusEdgeCells));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zminus)==Neighbor || getFaceCellIterator(zminus,"plusEdgeCells")==getFaceIterator__New(zminus,ExtraPlusEdgeCells));
  
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getFaceCellIterator(xplus,"NC_vars")==getFaceIterator__New(xplus,FaceNodes));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getFaceCellIterator(xminus,"NC_vars")==getFaceIterator__New(xminus,FaceNodes));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yplus)==Neighbor || getFaceCellIterator(yplus,"NC_vars")==getFaceIterator__New(yplus,FaceNodes));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yminus)==Neighbor || getFaceCellIterator(yminus,"NC_vars")==getFaceIterator__New(yminus,FaceNodes));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zplus)==Neighbor || getFaceCellIterator(zplus,"NC_vars")==getFaceIterator__New(zplus,FaceNodes));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(zminus)==Neighbor || getFaceCellIterator(zminus,"NC_vars")==getFaceIterator__New(zminus,FaceNodes));

  /*********These don't line up I think original FC_vars was bugged*********************
  ASSERT(getBCType(xplus)==Neighbor || getFaceCellIterator(xplus,"FC_vars")==getFaceIterator__New(xplus,SFCVars));
  ASSERT(getBCType(xminus)==Neighbor || getFaceCellIterator(xminus,"FC_vars")==getFaceIterator__New(xminus,SFCvars));
  ASSERT(getBCType(yplus)==Neighbor || getFaceCellIterator(yplus,"FC_vars")==getFaceIterator__New(yplus,SFCvars));
  ASSERT(getBCType(yminus)==Neighbor || getFaceCellIterator(yminus,"FC_vars")==getFaceIterator__New(yminus,SFCvars));
  ASSERT(getBCType(zplus)==Neighbor || getFaceCellIterator(zplus,"FC_vars")==getFaceIterator__New(zplus,SFCvars));
  ASSERT(getBCType(zminus)==Neighbor || getFaceCellIterator(zminus,"FC_vars")==getFaceIterator__New(zminus,SFCvars));
  */

  
  ASSERT(getBCType(xplus)==Neighbor || getFaceCellIterator(xplus,"alongInteriorFaceCells")==getFaceIterator__New(xplus,InteriorFaceCells));
  ASSERT(getBCType(xminus)==Neighbor || getFaceCellIterator(xminus,"alongInteriorFaceCells")==getFaceIterator__New(xminus,InteriorFaceCells));
  ASSERT(getBCType(yplus)==Neighbor || getFaceCellIterator(yplus,"alongInteriorFaceCells")==getFaceIterator__New(yplus,InteriorFaceCells));
  ASSERT(getBCType(yminus)==Neighbor || getFaceCellIterator(yminus,"alongInteriorFaceCells")==getFaceIterator__New(yminus,InteriorFaceCells));
  ASSERT(getBCType(zplus)==Neighbor || getFaceCellIterator(zplus,"alongInteriorFaceCells")==getFaceIterator__New(zplus,InteriorFaceCells));
  ASSERT(getBCType(zminus)==Neighbor || getFaceCellIterator(zminus,"alongInteriorFaceCells")==getFaceIterator__New(zminus,InteriorFaceCells));

/*
  FaceType face0=xminus,face1=yminus;
  if( !( getBCType(face0)==Neighbor || getBCType(face1)==Neighbor || getEdgeCellIterator(face0,face1)==getEdgeCellIterator__New(face0,face1) ) )
  {
    cout << "old:" << getEdgeCellIterator(face0,face1).begin() << " -> " << getEdgeCellIterator(face0,face1).end() << endl;
    cout << "new:" << getEdgeCellIterator__New(face0,face1).begin() << "->" << getEdgeCellIterator__New(face0,face1).end() << endl;

  }
*/

  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getBCType(yminus)==Neighbor ||  getEdgeCellIterator(xminus,yminus)==getEdgeCellIterator__New(xminus,yminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getBCType(yplus)==Neighbor ||  getEdgeCellIterator(xminus,yplus)==getEdgeCellIterator__New(xminus,yplus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getBCType(zminus)==Neighbor ||  getEdgeCellIterator(xminus,zminus)==getEdgeCellIterator__New(xminus,zminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xminus)==Neighbor || getBCType(zplus)==Neighbor ||  getEdgeCellIterator(xminus,zplus)==getEdgeCellIterator__New(xminus,zplus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getBCType(yminus)==Neighbor ||  getEdgeCellIterator(xplus,yminus)==getEdgeCellIterator__New(xplus,yminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getBCType(yplus)==Neighbor ||  getEdgeCellIterator(xplus,yplus)==getEdgeCellIterator__New(xplus,yplus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getBCType(zminus)==Neighbor ||  getEdgeCellIterator(xplus,zminus)==getEdgeCellIterator__New(xplus,zminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(xplus)==Neighbor || getBCType(zplus)==Neighbor ||  getEdgeCellIterator(xplus,zplus)==getEdgeCellIterator__New(xplus,zplus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yminus)==Neighbor || getBCType(zminus)==Neighbor ||  getEdgeCellIterator(yminus,zminus)==getEdgeCellIterator__New(yminus,zminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yminus)==Neighbor || getBCType(zplus)==Neighbor ||  getEdgeCellIterator(yminus,zplus)==getEdgeCellIterator__New(yminus,zplus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yplus)==Neighbor || getBCType(zminus)==Neighbor ||  getEdgeCellIterator(yplus,zminus)==getEdgeCellIterator__New(yplus,zminus));
  ASSERT(d_extraCells!=IntVector(1,1,1) || getBCType(yplus)==Neighbor || getBCType(zplus)==Neighbor ||  getEdgeCellIterator(yplus,zplus)==getEdgeCellIterator__New(yplus,zplus));

  ASSERT(d_grid[d_patchState.gridIndex]!=0);
  ASSERT(getLevel()==getLevel__New());

#endif 
}

int Patch::getGridIndex() const 
{
  int index = d_level_index;
  int levelid = d_level->getIndex();
  GridP grid = d_level->getGrid();

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
                            {getExtraCellLowIndex__New(),getCellLowIndex__New()},  //x-,y-,z- corner
                            {getCellHighIndex__New(),getExtraCellHighIndex__New()} //x+,y+,z+ corner
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

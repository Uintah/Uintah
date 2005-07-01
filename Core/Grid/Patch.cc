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

// This guy is bad news for SCIRun type applications that make use of
// multiple Grids that all access the same DataArchive whose patches
// all have hard coded IDs.
// 
// This variable should be unique to each Grid allocated by SCIRun,
// however, it is used in a static function (getByID) that has no
// knowledge of which grid a patch could be in.  Fortunately getByID
// is not used by SCIRun functions and should never be used anywhere
// where you could have multiple Grids that access a DataArchive.
// 
// The point I'm trying to make, is this global variable can get you
// into trouble when you use Patches from the DataArchive instead of
// the simulation.
// 
static map<int, const Patch*> patchIDtoPointerMap;
static Mutex patchIDtoPointerMap_lock("patchIDtoPointerMap lock");

static AtomicCounter* ids = 0;
static Mutex ids_init("ID init");

Patch::Patch(const Level* level,
	     const IntVector& lowIndex, const IntVector& highIndex,
	     const IntVector& inLowIndex, const IntVector& inHighIndex,
	     int id)
    : d_realPatch(0), d_level(level), d_level_index(-1),
      d_lowIndex(lowIndex), d_highIndex(highIndex),
      d_inLowIndex(inLowIndex), d_inHighIndex(inHighIndex),
      d_id( id )
{
  have_layout=false;
  if(!ids){
    ids_init.lock();
    if(!ids){
      ids = new AtomicCounter("Patch ID counter", 0);
    }
    ids_init.unlock();
    
  }
  if(d_id == -1){
    d_id = (*ids)++;

    patchIDtoPointerMap_lock.lock();
    if(patchIDtoPointerMap.find(d_id) != patchIDtoPointerMap.end()){
      cerr << "id=" << d_id << '\n';
      SCI_THROW(InternalError("duplicate patch!"));
    }
    patchIDtoPointerMap[d_id]=this;
    patchIDtoPointerMap_lock.unlock();
    in_database=true;
  } else {
    in_database=false;
    if(d_id >= *ids)
      ids->set(d_id+1);
   }
  setBCType(xminus, None);
  setBCType(xplus, None);
  setBCType(yminus, None);
  setBCType(yplus, None);
  setBCType(zminus, None);
  setBCType(zplus, None);
  
  d_hasCoarsefineInterfaceFace = false;

  d_nodeHighIndex = d_highIndex+
    IntVector(getBCType(xplus) == Neighbor?0:1,
	      getBCType(yplus) == Neighbor?0:1,
	      getBCType(zplus) == Neighbor?0:1);
  
}

Patch::Patch(const Patch* realPatch, const IntVector& virtualOffset)
    : d_realPatch(realPatch), d_level(realPatch->d_level),
      d_level_index(realPatch->d_level_index),
      d_lowIndex(realPatch->d_lowIndex + virtualOffset),
      d_highIndex(realPatch->d_highIndex + virtualOffset),
      d_inLowIndex(realPatch->d_inLowIndex + virtualOffset),
      d_inHighIndex(realPatch->d_inHighIndex + virtualOffset),
      d_nodeHighIndex(realPatch->d_nodeHighIndex + virtualOffset),
      array_bcs(realPatch->array_bcs),
      have_layout(realPatch->have_layout),
      layouthint(realPatch->layouthint)
{
  //if(!ids){
  // make the id be -1000 * realPatch id - some first come, first serve index
  d_id = -1000 * realPatch->d_id; // temporary
  int index = 1;
  // Since we can have multiple grids adding their patches to the
  // patchIDtoPointerMap variable, we need to count only those patches that
  // actually match realPatch.
  int numVirtualPatches = 0;
  map<int, const Patch*>::iterator iter;
  // Need to lock this, so it will be thread safe
  patchIDtoPointerMap_lock.lock();    
  while ((iter = patchIDtoPointerMap.find(d_id - index)) !=
         patchIDtoPointerMap.end()){
    ++index;
    // Check to see if this is one of our patches
    if (d_realPatch == iter->second->getRealPatch())
      if (++numVirtualPatches >= 27)
        SCI_THROW(InternalError("A real patch shouldn't have more than 26 (3*3*3 - 1) virtual patches"));
  }
  d_id -= index;
  // Double check to make sure that the patch has not already been added
  ASSERT(patchIDtoPointerMap.find(d_id) == patchIDtoPointerMap.end());    
  patchIDtoPointerMap[d_id]=this;
  patchIDtoPointerMap_lock.unlock();    
  in_database = true;
  //}      
  
  for (int i = 0; i < numFaces; i++)
    d_bctypes[i] = realPatch->d_bctypes[i];
}

Patch::~Patch()
{
  d_BoundaryFaces.clear();
  d_coarseFineInterfaceFaces.clear();

  for (FaceType face = startFace; face <= endFace; face = nextFace(face))
    d_CornerCells[face].clear();

  if(in_database){
//     patchIDtoPointerMap.erase( patchIDtoPointerMap.find(getID()));
    patchIDtoPointerMap_lock.lock();
    patchIDtoPointerMap.erase( getID() );
    patchIDtoPointerMap_lock.unlock();
  }
 for(Patch::FaceType face = Patch::startFace;
     face <= Patch::endFace; face=Patch::nextFace(face))
    delete array_bcs[face];

  array_bcs.clear();
}

const Patch* Patch::getByID(int id)
{
  map<int, const Patch*>::iterator iter = patchIDtoPointerMap.find(id);
  if(iter == patchIDtoPointerMap.end())
    return 0;
  else
    return iter->second;
}

Vector Patch::dCell() const {
  // This will need to change for stretched grids
  return d_level->dCell();
}

Point Patch::nodePosition(const IntVector& idx) const {
  return d_level->getNodePosition(idx);
}

Point Patch::cellPosition(const IntVector& idx) const {
  return d_level->getCellPosition(idx);
}

int Patch::findClosestNode(const Point& pos, IntVector& idx) const
{
  int p[3];
  idx = d_level->getCellIndex(pos);
  Point cellP = d_level->getCellPosition(idx);
  for(int i=0;i<3;++i) {
    if( pos(i)>cellP(i) ) {
      idx[i]++;
      p[i] = 1;
    }
    else p[i] = 0;
  }
  return p[0]+p[1]*2+p[2]*4;
}

bool Patch::findCell(const Point& pos, IntVector& ci) const
{
   ci = d_level->getCellIndex(pos);
   return containsCell(ci);
}

void Patch::findCellsFromNode( const IntVector& nodeIndex,
                               IntVector cellIndex[8]) const
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
                               IntVector nodeIndex[8]) const
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



// for Fracture below **************************************
void Patch::findCellNodes(const Point& pos,
                               IntVector ni[8]) const
{
   Point cellpos = d_level->positionToIndex(pos);
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

void Patch::findCellNodes27(const Point& pos,
                                 IntVector ni[27]) const
{
   Point cellpos = d_level->positionToIndex(pos);
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
// for Fracture above *******************************************


ostream& operator<<(ostream& out, const Patch & r)
{
  out.setf(ios::scientific,ios::floatfield);
  out.precision(4);
  out << "(Patch " << r.getID() << ": box=" << r.getBox()
      << ", lowIndex=" << r.getCellLowIndex() << ", highIndex=" 
      << r.getCellHighIndex() << ")";
  out.setf(ios::scientific ,ios::floatfield);
  return out;
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
    SCI_THROW(InvalidGrid( msg.str() ));
  }
}

Patch::BCType 
Patch::getBCType(Patch::FaceType face) const
{
  return d_bctypes[face];
}

void
Patch::setBCType(Patch::FaceType face, BCType newbc)
{
   d_bctypes[face]=newbc;
   d_nodeHighIndex = d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			   getBCType(yplus) == Neighbor?0:1,
			   getBCType(zplus) == Neighbor?0:1);


   // If this face has a BCType of Patch::None, make sure
   // that it is in the list of d_BoundaryFaces, otherwise, make
   // sure that it is not in this list.
   
   vector<FaceType>::iterator faceIdx = d_BoundaryFaces.begin();
   vector<FaceType>::iterator faceEnd = d_BoundaryFaces.end();

   while (faceIdx != faceEnd) {
     if (*faceIdx == face) break;
     faceIdx++;
   }

   if (newbc == Patch::None) {
    if(faceIdx == d_BoundaryFaces.end()){
     d_BoundaryFaces.push_back(face);
    }
   } else {
     if (faceIdx != d_BoundaryFaces.end()) {
       d_BoundaryFaces.erase(faceIdx);
     }
   }
   
   
   //__________________________________
   //  set the coarse fine interface faces
   vector<FaceType>::iterator face_Idx = d_coarseFineInterfaceFaces.begin();
   vector<FaceType>::iterator face_End = d_coarseFineInterfaceFaces.end();

   while (face_Idx != face_End) {
     if (*face_Idx == face) break;
     face_Idx++;
   }

   if (newbc == Patch::Coarse ) {
     if(face_Idx == d_coarseFineInterfaceFaces.end()){  
       d_coarseFineInterfaceFaces.push_back(face);
       d_hasCoarsefineInterfaceFace = true;
     }
   } else {
     if (face_Idx != d_coarseFineInterfaceFaces.end()) {
       d_coarseFineInterfaceFaces.erase(face_Idx);
     }
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
Patch::getArrayBCValues(Patch::FaceType face,int mat_id,string type,
			vector<IntVector>& bound, 
			vector<IntVector>& nbound,
			vector<IntVector>& sfx, 
			vector<IntVector>& sfy, 
			vector<IntVector>& sfz,
			int child) const
{
  map<Patch::FaceType,BCDataArray* >::const_iterator itr=array_bcs.find(face); 
  if (itr != array_bcs.end()) {
    const BoundCondBase* bc = itr->second->getBoundCondData(mat_id,type,child);
    itr->second->getBoundaryIterator(mat_id,bound,child);
    itr->second->getNBoundaryIterator(mat_id,nbound,child);
    itr->second->getSFCXIterator(mat_id,sfx,child);
    itr->second->getSFCYIterator(mat_id,sfy,child);
    itr->second->getSFCZIterator(mat_id,sfz,child);
    return bc;
  } else
    return 0;
}

bool 
Patch::haveBC(FaceType face,int mat_id,string bc_type,string bc_variable) const
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
     SCI_THROW(InternalError("Illegal FaceType in Patch::getFace"));
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::faceDirection"));
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::faceName"));
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
    SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceNodes"));
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
     SCI_THROW(InternalError("Illegal FaceType in Patch::getFaceNodes"));
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

void Patch::setExtraIndices(const IntVector& l, const IntVector& h)
{
  d_lowIndex = l;
  d_highIndex = h;
}

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
    SCI_THROW(InternalError("Patch::getSFCIterator: dir must be 0, 1, or 2"));
  }
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
    
    const vector<IntVector> corner = getCornerCells(face0);
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

//__________________________________
//  Returns the principal axis along a face and
//  the orthognonal axes to that face (right hand rule).
IntVector
Patch::faceAxes(const FaceType& face) const
{
  IntVector dir(0,0,0);
  if (face == xminus || face == xplus ) {
    dir = IntVector(0,1,2);
  }
  if (face == yminus || face == yplus ) {
    dir = IntVector(1,2,0);
  }
  if (face == zminus || face == zplus ) {
    dir = IntVector(2,0,1);
  }
  return dir;
}
//______________________________________________________________________


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

NodeIterator Patch::getNodeIterator(int n8or27) const
{
  if(n8or27!=27){
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
  return d_lowIndex - IntVector(d_bctypes[xminus] == Neighbor?numGC:0,
				d_bctypes[yminus] == Neighbor?numGC:0,
				d_bctypes[zminus] == Neighbor?numGC:0);

}

IntVector Patch::getGhostCellHighIndex(int numGC) const
{
  return d_highIndex + IntVector(d_bctypes[xplus] == Neighbor?numGC:0,
				 d_bctypes[yplus] == Neighbor?numGC:0,
				 d_bctypes[zplus] == Neighbor?numGC:0);
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
    SCI_THROW(InternalError("ghost cells should not be specified with Ghost::None"));

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
      SCI_THROW(InternalError(basisName + " around " + ghostTypeName + " not supported for ghost offsets"));
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

void Patch::computeVariableExtents(TypeDescription::Type basis,
				   const IntVector& boundaryLayer,
				   Ghost::GhostType gtype, int numGhostCells,
				   IntVector& low, IntVector& high) const
{
  bool basisMustExist = (gtype != Ghost::None);
  computeVariableExtents(translateTypeToBasis(basis, basisMustExist),
			 boundaryLayer, gtype, numGhostCells, low, high);
}

void Patch::computeVariableExtents(TypeDescription::Type basis,
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
				 Patch::selectType& selected_patches) const
{
  ASSERT(levelOffset == 1 || levelOffset == -1);

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
  //cout << "  Patch:Golp: " << low << " " << high << endl;
  Level::selectType patches;
  otherLevel->selectPatches(low, high, patches); 
  
  // based on the expanded range above to search for extra cells, we might
  // have grabbed more patches than we wanted, so refine them here
  
  for (int i = 0; i < patches.size(); i++) {
    if (levelOffset < 0 || getBox().overlaps(patches[i]->getBox())) {
      selected_patches.push_back(patches[i]);
    }
  }
}

Patch::VariableBasis Patch::translateTypeToBasis(TypeDescription::Type type,
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
    return CellBased;
  default:
    if (mustExist)
      SCI_THROW(InternalError("Unknown variable type in Patch::getVariableExtents (from TypeDescription::Type)"));
    else
      return CellBased; // doesn't matter
  }
}

void Patch::setLayoutHint(const IntVector& pos)
{
  ASSERT(!have_layout);
  layouthint = pos;
  have_layout=true;
}

bool Patch::getLayoutHint(IntVector& pos) const
{
  pos = layouthint;
  return have_layout;
}

Box Patch::getBox() const {
  return d_level->getBox(d_lowIndex, d_highIndex);
}

IntVector Patch::neighborsLow() const
{
  return IntVector(d_bctypes[xminus] == Neighbor? 0:1,
		   d_bctypes[yminus] == Neighbor? 0:1,
		   d_bctypes[zminus] == Neighbor? 0:1);
}

IntVector Patch::neighborsHigh() const
{
  return IntVector(d_bctypes[xplus] == Neighbor? 0:1,
		   d_bctypes[yplus] == Neighbor? 0:1,
		   d_bctypes[zplus] == Neighbor? 0:1);
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
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)"));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)"));
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
    SCI_THROW(InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)"));
  default:
    SCI_THROW(InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)"));
  }
}


void Patch::finalizePatch()
{
  //////////
  // Calculate with of this patche's cells are on the corner
  // of the domain and keep a list of these cells for each
  // face of the patch.

  IntVector low,hi;
  low = getLowIndex();
  hi  = getHighIndex() - IntVector(1,1,1);  

  IntVector patchNeighborLow  = neighborsLow();
  IntVector patchNeighborHigh = neighborsHigh();

  vector<FaceType>::const_iterator iter;
  for (iter  = getBoundaryFaces()->begin(); 
       iter != getBoundaryFaces()->end(); ++iter){
    FaceType face = *iter;  
 
    IntVector axes = faceAxes(face);
    int P_dir = axes[0];  // principal direction
    int dir1  = axes[1];  // other vector directions
    int dir2  = axes[2]; 

    //__________________________________
    // main index for that face plane
    int plusMinus = faceDirection(face)[P_dir];
    int main_index = 0;
    if( plusMinus == 1 ) { // plus face
      main_index = hi[P_dir];
    } else {               // minus faces
      main_index = low[P_dir];
    }

    //__________________________________
    // Looking down on the face examine 
    // each corner (clockwise) and if there
    // are no neighboring patches then set the
    // index
    // 
    // Top-right corner
    IntVector corner(-9,-9,-9);
    if ( patchNeighborHigh[dir1] == 1 && patchNeighborHigh[dir2] == 1) {
      corner[P_dir] = main_index;
      corner[dir1]  = hi[dir1];
      corner[dir2]  = hi[dir2];
      d_CornerCells[face].push_back(corner);
    }
    // bottom-right corner
    if ( patchNeighborLow[dir1] == 1 && patchNeighborHigh[dir2] == 1) {
      corner[P_dir] = main_index;
      corner[dir1]  = low[dir1];
      corner[dir2]  = hi[dir2];
      d_CornerCells[face].push_back(corner);
    } 
    // bottom-left corner
    if ( patchNeighborLow[dir1] == 1 && patchNeighborLow[dir2] == 1) {
      corner[P_dir] = main_index;
      corner[dir1]  = low[dir1];
      corner[dir2]  = low[dir2];
      d_CornerCells[face].push_back(corner);
    } 
    // Top-left corner
    if ( patchNeighborHigh[dir1] == 1 && patchNeighborLow[dir2] == 1) {
      corner[P_dir] = main_index;
      corner[dir1]  = hi[dir1];
      corner[dir2]  = low[dir2];
      d_CornerCells[face].push_back(corner);
    } 
  }
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

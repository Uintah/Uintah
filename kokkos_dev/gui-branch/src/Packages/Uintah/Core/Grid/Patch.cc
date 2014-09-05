
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Math/Primes.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>

#include <Core/Thread/AtomicCounter.h>
#include <Core/Math/MiscMath.h>

#include <values.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <map>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

static map<int, const Patch*> patches;

static AtomicCounter* ids = 0;
static Mutex ids_init("ID init");

Patch::Patch(const Level* level,
	     const IntVector& lowIndex, const IntVector& highIndex,
	     const IntVector& inLowIndex, const IntVector& inHighIndex,
	     int id)
    : d_level(level), d_level_index(-1),
      d_lowIndex(lowIndex), d_highIndex(highIndex),
      d_inLowIndex(inLowIndex), d_inHighIndex(inHighIndex),
      d_id( id )
{
  have_layout=false;
  if(d_id == -1){
    if(!ids){
      ids_init.lock();
      if(!ids){
	ids = new AtomicCounter("Patch ID counter", 0);
      }
      ids_init.unlock();
      
    }
    d_id = (*ids)++;

    if(patches.find(d_id) != patches.end()){
      cerr << "id=" << d_id << '\n';
      throw InternalError("duplicate patch!");
    }
    patches[d_id]=this;
    in_database=true;
   } else {
     in_database=false;
   }

   d_bcs.resize(numFaces);

   d_nodeHighIndex = d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1);
}

Patch::~Patch()
{
  if(in_database)
    patches.erase(patches.find(getID()));
}

const Patch* Patch::getByID(int id)
{
  map<int, const Patch*>::iterator iter = patches.find(id);
  if(iter == patches.end())
    return 0;
  else
    return iter->second;
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

void Patch::findCellAndWeights(const Point& pos,
				IntVector ni[8], double S[8]) const
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
   double fx = cellpos.x() - ix;
   double fy = cellpos.y() - iy;
   double fz = cellpos.z() - iz;
   double fx1 = 1-fx;
   double fy1 = 1-fy;
   double fz1 = 1-fz;
   S[0] = fx1 * fy1 * fz1;
   S[1] = fx1 * fy1 * fz;
   S[2] = fx1 * fy * fz1;
   S[3] = fx1 * fy * fz;
   S[4] = fx * fy1 * fz1;
   S[5] = fx * fy1 * fz;
   S[6] = fx * fy * fz1;
   S[7] = fx * fy * fz;
}


void Patch::findCellAndShapeDerivatives(const Point& pos,
					 IntVector ni[8],
					 Vector d_S[8]) const
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
   double fx = cellpos.x() - ix;
   double fy = cellpos.y() - iy;
   double fz = cellpos.z() - iz;
   double fx1 = 1-fx;
   double fy1 = 1-fy;
   double fz1 = 1-fz;
   d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
   d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
   d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
   d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
   d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
   d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
   d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
   d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

void Patch::findCellAndWeightsAndShapeDerivatives(const Point& pos,
					          IntVector ni[8],
						  double S[8],
					          Vector d_S[8]) const
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
   double fx = cellpos.x() - ix;
   double fy = cellpos.y() - iy;
   double fz = cellpos.z() - iz;
   double fx1 = 1-fx;
   double fy1 = 1-fy;
   double fz1 = 1-fz;
   S[0] = fx1 * fy1 * fz1;
   S[1] = fx1 * fy1 * fz;
   S[2] = fx1 * fy * fz1;
   S[3] = fx1 * fy * fz;
   S[4] = fx * fy1 * fz1;
   S[5] = fx * fy1 * fz;
   S[6] = fx * fy * fz1;
   S[7] = fx * fy * fz;
   d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
   d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
   d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
   d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
   d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
   d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
   d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
   d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

ostream& operator<<(ostream& out, const Patch & r)
{
  out << "(Patch " << r.getID() << ": box=" << r.getBox()
      << ", lowIndex=" << r.getCellLowIndex() << ", highIndex=" 
      << r.getCellHighIndex() << ")";
  return out;
}

long Patch::totalCells() const
{
   IntVector res(d_highIndex-d_lowIndex);
   return res.x()*res.y()*res.z();
}

void Patch::performConsistencyCheck() const
{
   IntVector res(d_highIndex-d_lowIndex);
   if(res.x() < 1 || res.y() < 1 || res.z() < 1) {
      ostringstream msg;
      msg << "Degenerate patch: " << toString() << " (resolution=" << res << ")";
      throw InvalidGrid( msg.str() );
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
}

void 
Patch::setBCValues(Patch::FaceType face, BCData& bc)
{
  d_bcs[face] = bc;
  d_nodeHighIndex = d_highIndex+
    IntVector(getBCType(xplus) == Neighbor?0:1,
	      getBCType(yplus) == Neighbor?0:1,
	      getBCType(zplus) == Neighbor?0:1);
}

BoundCondBase*
Patch::getBCValues(int mat_id,string type,Patch::FaceType face) const
{
  return d_bcs[face].getBCValues(mat_id,type);
}


void
Patch::getFace(FaceType face, int offset, IntVector& l, IntVector& h) const
{
   l=getCellLowIndex();
   h=getCellHighIndex();
   switch(face){
   case xminus:
      l.x(l.x()-offset);
      h.x(l.x()+2-offset);
      break;
   case xplus:
      l.x(h.x()-1+offset);
      h.x(h.x()+offset);
      break;
   case yminus:
      l.y(l.y()-offset);
      h.y(l.y()+2-offset);
      break;
   case yplus:
      l.y(h.y()-1+offset);
      h.y(h.y()+offset);
      break;
   case zminus:
      l.z(l.z()-offset);
      h.z(l.z()+2-offset);
      break;
   case zplus:
      l.z(h.z()-1+offset);
      h.z(h.z()+offset);
      break;
   default:
       throw InternalError("Illegal FaceType in Patch::getFace");
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

CellIterator
Patch::getCellIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low((int)l.x(), (int)l.y(), (int)l.z());
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
   low = Max(low, getCellLowIndex());
   high = Min(high, getCellHighIndex());
   return CellIterator(low, high);
}
CellIterator
Patch::getExtraCellIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low((int)l.x(), (int)l.y(), (int)l.z());
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
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


//__________________________________
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


CellIterator    
Patch::getFaceCellIterator(const FaceType& face, const string& domain) const
{ 
  // Iterate over the GhostCells on a particular face
  // if domain == plusEdgeCells this includes the edge and corner cells.
  // T.Harman
  
  IntVector lowPt  = d_inLowIndex;
  IntVector highPt = d_inHighIndex;
  int offset = 0;
  if(domain == "plusEdgeCells"){
    lowPt   = d_lowIndex;
    highPt  = d_highIndex;
  }
  // This will allow you to hit all the nodes/faces on the border of
  // the extracells
  if(domain == "NC_vars"|| domain == "FC_vars"){  
    lowPt   = d_inLowIndex;
    highPt  = d_highIndex;
    offset  = 1;
  }
  if(domain == "FC_vars"){  
    lowPt   = d_lowIndex;
    highPt  = d_highIndex;
    offset  = 1;
  }

  if (face == Patch::xplus) {           //  X P L U S
     lowPt.x(d_inHighIndex.x());
     highPt.x(d_inHighIndex.x()+1);
  }
  if(face == Patch::xminus){            //  X M I N U S
    highPt.x(d_inLowIndex.x()  + offset);
    lowPt.x(d_inLowIndex.x()-1 + offset);
  }
  if(face == Patch::yplus) {            //  Y P L U S
    lowPt.y(d_inHighIndex.y());
    highPt.y(d_inHighIndex.y()+1);
  }
  if(face == Patch::yminus) {           //  Y M I N U S
    highPt.y(d_inLowIndex.y()  + offset);
    lowPt.y(d_inLowIndex.y()-1 + offset);
  }
  if (face == Patch::zplus) {           //  Z P L U S
    lowPt.z(d_inHighIndex.z() );
    highPt.z(d_inHighIndex.z()+1);
  }
  if (face == Patch::zminus) {          //  Z M I N U S
    highPt.z(d_inLowIndex.z()  + offset);
    lowPt.z(d_inLowIndex.z()-1 + offset);
  } 
  return CellIterator(lowPt, highPt);
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

NodeIterator
Patch::getNodeIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low((int)l.x(), (int)l.y(), (int)l.z());
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
   low = Max(low, getNodeLowIndex());
   high = Min(high, getNodeHighIndex());
   return NodeIterator(low, high);
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
    throw InternalError("ghost cells should not be specified with Ghost::None");

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
    else if (basis == gtype) {
      // nodes around nodes or faces around faces
      lowOffset = highOffset = g * dir; 
    }
    else if (gtype == Ghost::AroundFaces) {
      lowOffset = highOffset = g;
    }
    else {
      string basisName = Ghost::getGhostTypeName((Ghost::GhostType)basis);
      string ghostTypeName = Ghost::getGhostTypeName(gtype);
      throw InternalError(basisName + " around " + ghostTypeName + " not supported for ghost offsets");
    }
  }

  ASSERT(lowOffset[0] >= 0 && lowOffset[1] >= 0 && lowOffset[2] >= 0 &&
	 highOffset[0] >= 0 && highOffset[2] >= 0 && highOffset[2] >= 0); 
}

void Patch::computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				   int numGhostCells,
				   IntVector& low, IntVector& high) const
{
  IntVector lowOffset, highOffset;
  getGhostOffsets(basis, gtype, numGhostCells, lowOffset, highOffset);
  computeExtents(basis, lowOffset, highOffset, low, high);
}

void Patch::computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				   int numGhostCells,
				   Level::selectType& neighbors,
				   IntVector& low, IntVector& high) const
{
  IntVector lowOffset, highOffset;
  getGhostOffsets(basis, gtype, numGhostCells, lowOffset, highOffset);
  computeExtents(basis, lowOffset, highOffset, low, high);
  d_level->selectPatches(low, high, neighbors);
#if 0
  IntVector low2;
  IntVector high2;
  Level::selectType neighbors2;
  computeVariableExtents2(basis, gtype, numGhostCells, neighbors2, low2, high2);
  ASSERT(low == low2);
  ASSERT(high == high2);
#endif
}


void Patch::computeExtents(VariableBasis basis, const IntVector& lowOffset,
			   const IntVector& highOffset,
			   IntVector& low, IntVector& high) const

{
  ASSERT(lowOffset[0] >= 0 && lowOffset[1] >= 0 && lowOffset[2] >= 0 &&
	 highOffset[0] >= 0 && highOffset[2] >= 0 && highOffset[2] >= 0);
  
  IntVector origLowIndex = getLowIndex(basis);
  IntVector origHighIndex = getHighIndex(basis);
  low = origLowIndex - lowOffset;
  high = origHighIndex + highOffset;

  for (int i = 0; i < 3; i++) {
    FaceType faceType = (FaceType)(2 * i); // x, y, or z minus
    if (getBCType(faceType) != Neighbor) {
      // no neighbor -- use original low index for that side
      low[i] = origLowIndex[i];
    }
    
    faceType = (FaceType)(faceType + 1); // x, y, or z plus
    if (getBCType(faceType) != Neighbor) {
      // no neighbor -- use original high index for that side
      high[i] = origHighIndex[i];
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
      throw InternalError("Unknown variable type in Patch::getVariableExtents (from TypeDescription::Type)");
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

//
// $Id$
//

#include <Uintah/Grid/Patch.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/SubPatch.h>
#include <Uintah/Math/Primes.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/AtomicCounter.h>

#include <values.h>
#include <iostream>
#include <sstream>

using SCICore::Exceptions::InternalError;
using SCICore::Geometry::Max;
using SCICore::Geometry::Min;
using SCICore::Math::Floor;
using namespace SCICore::Geometry;
using namespace Uintah;
using namespace std;
static SCICore::Thread::AtomicCounter ids("Patch ID counter", 0);

Patch::Patch(const Level* level,
	     const IntVector& lowIndex, const IntVector& highIndex,
	     int id)
    : d_level(level), d_lowIndex(lowIndex), d_highIndex(highIndex),
      d_id( id )
{
   if(d_id == -1)
      d_id = ids++;

   d_bcs = vector<vector<BoundCond* > >(numFaces);
   for (int i = 0; i<numFaces; i++ ) {
     vector<BoundCond* > a;
     d_bcs[i] = a;
   }
}

Patch::~Patch()
{
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

bool Patch::findCellAndWeights(const Point& pos,
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
   return ix>= d_lowIndex.x()-1 && iy>=d_lowIndex.y()-1 && iz>=d_lowIndex.z()-1 && ix<d_highIndex.x() && iy<d_highIndex.y() && iz<d_highIndex.z();
}


bool Patch::findCellAndShapeDerivatives(const Point& pos,
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
   return ix>= d_lowIndex.x()-1 && iy>=d_lowIndex.y()-1 && iz>=d_lowIndex.z()-1 && ix<d_highIndex.x() && iy<d_highIndex.y() && iz<d_highIndex.z();
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
}

void 
Patch::setBCValues(Patch::FaceType face, vector<BoundCond*>& bc)
{
  d_bcs[face] = bc;
}

vector<BoundCond* >
Patch::getBCValues(Patch::FaceType face) const
{
  return d_bcs[face];
}


void
Patch::getFace(FaceType face, int offset, IntVector& l, IntVector& h) const
{
   l=getCellLowIndex();
   h=getCellHighIndex();
   //   std::cout << "cell low index = " << l << " hi index = " << h << endl;
   switch(face){
   case xminus:
      l.x(l.x()-offset);
      h.x(l.x()+2-offset);
      break;
   case xplus:
     //      l.x(h.x()-1+offset);
      l.x(h.x()-1+offset);
      //      h.x(h.x()+offset);
      h.x(h.x()+offset);
      break;
   case yminus:
      l.y(l.y()-offset);
      h.y(l.y()+2-offset);
      break;
   case yplus:
     //      l.y(h.y()-1+offset);
      l.y(h.y()-1+offset);
      //      h.y(h.y()+offset);
      h.y(h.y()+offset);
      break;
   case zminus:
      l.z(l.z()-offset);
      h.z(l.z()+2-offset);
      break;
   case zplus:
     //      l.z(h.z()-1+offset);
      l.z(h.z()-1+offset);
      //      h.z(h.z()+offset);
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

CellIterator
Patch::getCellIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low((int)l.x(), (int)l.y(), (int)l.z());
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
   low = SCICore::Geometry::Max(low, getCellLowIndex());
   high = SCICore::Geometry::Min(high, getCellHighIndex());
   return CellIterator(low, high);
}
CellIterator
Patch::getCellIterator() const
{
   return CellIterator(getCellLowIndex(), getCellHighIndex());
}

#if 0
Box Patch::getGhostBox(const IntVector& lowOffset,
		       const IntVector& highOffset) const
{
   return Box(d_level->getNodePosition(d_lowIndex+lowOffset),
	      d_level->getNodePosition(d_highIndex+highOffset));
}
#endif

NodeIterator Patch::getNodeIterator() const
{
   return NodeIterator(getNodeLowIndex(), getNodeHighIndex());
}

NodeIterator
Patch::getNodeIterator(const Box& b) const
{
   Point l = d_level->positionToIndex(b.lower());
   Point u = d_level->positionToIndex(b.upper());
   IntVector low((int)l.x(), (int)l.y(), (int)l.z());
   IntVector high(RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
   low = SCICore::Geometry::Max(low, getNodeLowIndex());
   high = SCICore::Geometry::Min(high, getNodeHighIndex());
   return NodeIterator(low, high);
}

IntVector Patch::getNodeHighIndex() const
{
   IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?0:1,
			 getBCType(yplus) == Neighbor?0:1,
			 getBCType(zplus) == Neighbor?0:1));
   return h;
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

void Patch::computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				   int numGhostCells,
				   vector<const Patch*>& neighbors,
				   IntVector& low, IntVector& high) const
{
    IntVector l(d_lowIndex);
    IntVector h(d_highIndex);
    IntVector g(numGhostCells, numGhostCells, numGhostCells);
    switch(basis){
    case CellBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // Cells around cells
	    l-=g;
	    h+=g;
	    break;
	case Ghost::AroundNodes:    // Cells around nodes
	    l-=g;
	    h+=g-IntVector(1,1,1);
	    break;
	case Ghost::AroundXFaces:   // Cells around x faces
	    l-=IntVector(numGhostCells,0,0);
	    h+=IntVector(numGhostCells-1,0,0);
	    break;
	case Ghost::AroundYFaces:   // Cells around y faces
	    l-=IntVector(0,numGhostCells,0);
	    h+=IntVector(0,numGhostCells-1,0);
	    break;
	case Ghost::AroundZFaces:   // Cells around z faces
	    l-=IntVector(0,0,numGhostCells);
	    h+=IntVector(0,0,numGhostCells-1);
	    break;
	case Ghost::AroundAllFaces: // Cells around all faces
	    l-=g;
	    h+=g-IntVector(1,1,1);
	    break;
	}
	break;
    case NodeBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // Nodes around cells
	    l-=g-IntVector(1,1,1);
	    h+=g;
	    break;
	case Ghost::AroundNodes:    // Nodes around nodes
	    l-=g;
	    h+=g;
	    break;
	case Ghost::AroundXFaces:   // Nodes around x faces
	    l-=IntVector(numGhostCells-1,0,0);
	    h+=IntVector(numGhostCells,0,0);
	    break;
	case Ghost::AroundYFaces:   // Nodes around y faces
	    l-=IntVector(0,numGhostCells-1,0);
	    h+=IntVector(0,numGhostCells,0);
	    break;
	case Ghost::AroundZFaces:   // Nodes around z faces
	    l-=IntVector(0,0,numGhostCells-1);
	    h+=IntVector(0,0,numGhostCells);
	    break;
	case Ghost::AroundAllFaces: // Nodes around all faces
	    l-=g-IntVector(1,1,1);
	    h+=g;
	    break;
	}
	break;
    case XFaceBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // X faces around cells
	    l-=IntVector(numGhostCells-1,0,0);
	    h+=IntVector(numGhostCells,0,0);
	    break;
	case Ghost::AroundNodes:    // X faces around nodes
	    throw InternalError("X faces around nodes not implemented");
	case Ghost::AroundXFaces:   // X faces around x faces
	    l-=IntVector(numGhostCells,0,0);
	    h+=IntVector(numGhostCells,0,0);
	    break;
	case Ghost::AroundYFaces:   // X faces around y faces
	    throw InternalError("X faces around y faces not implemented");
	case Ghost::AroundZFaces:   // X faces around z faces
	    throw InternalError("X faces around z faces not implemented");
	case Ghost::AroundAllFaces: // X faces around all faces
	    throw InternalError("X faces around all faces not implemented");
	    break;
	}
	break;
    case YFaceBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // Y faces around cells
	    l-=IntVector(0,numGhostCells-1,0);
	    h+=IntVector(0,numGhostCells,0);
	    break;
	case Ghost::AroundNodes:    // Y faces around nodes
	    throw InternalError("Y faces around nodes not implemented");
	case Ghost::AroundXFaces:   // Y faces around x faces
	    throw InternalError("Y faces around x faces not implemented");
	case Ghost::AroundYFaces:   // Y faces around y faces
	    l-=IntVector(0,numGhostCells,0);
	    h+=IntVector(0,numGhostCells,0);
	    break;
	case Ghost::AroundZFaces:   // Y faces around z faces
	    throw InternalError("Y faces around z faces not implemented");
	case Ghost::AroundAllFaces: // Y faces around all faces
	    throw InternalError("Y faces around all faces not implemented");
	    break;
	}
	break;
    case ZFaceBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // Z faces around cells
	    l-=IntVector(0,0,numGhostCells-1);
	    h+=IntVector(0,0,numGhostCells);
	    break;
	case Ghost::AroundNodes:    // Z faces around nodes
	    throw InternalError("Z faces around nodes not implemented");
	case Ghost::AroundXFaces:   // Z faces around x faces
	    throw InternalError("Z faces around x faces not implemented");
	case Ghost::AroundYFaces:   // Z faces around y faces
	    throw InternalError("Z faces around y faces not implemented");
	case Ghost::AroundZFaces:   // Z faces around z faces
	    l-=IntVector(0,0,numGhostCells);
	    h+=IntVector(0,0,numGhostCells);
	    break;
	case Ghost::AroundAllFaces: // Z faces around all faces
	    throw InternalError("Z faces around all faces not implemented");
	    break;
	}
	break;
    case AllFaceBased:
	switch(gtype){
	case Ghost::None:
	    if(gtype != 0)
		throw InternalError("ghost cells should not be specified with Ghost::None");
	    break;
	case Ghost::AroundCells:    // All faces around cells
	    throw InternalError("All faces around cells not implemented");
	case Ghost::AroundNodes:    // All faces around nodes
	    throw InternalError("All faces around nodes not implemented");
	case Ghost::AroundXFaces:   // All faces around x faces
	    throw InternalError("All faces around x faces not implemented");
	case Ghost::AroundYFaces:   // All faces around y faces
	    throw InternalError("All faces around y faces not implemented");
	case Ghost::AroundZFaces:   // All faces around z faces
	    throw InternalError("All faces around z faces not implemented");
	case Ghost::AroundAllFaces: // All faces around all faces
	    throw InternalError("All faces around all faces not implemented");
	    break;
	}
	break;
    }
    d_level->selectPatches(l, h, neighbors);
}

void Patch::computeVariableExtents(TypeDescription::Type basis,
				   Ghost::GhostType gtype,
				   int numGhostCells,
				   vector<const Patch*>& neighbors,
				   IntVector& low, IntVector& high) const
{
    VariableBasis translation;
    switch(basis){
    case TypeDescription::CCVariable:
	translation=CellBased;
	break;
    case TypeDescription::NCVariable:
	translation=NodeBased;
	break;
    case TypeDescription::FCVariable:
	translation=AllFaceBased;
	break;
    case TypeDescription::SFCXVariable:
	translation=XFaceBased;
	break;
    case TypeDescription::SFCYVariable:
	translation=YFaceBased;
	break;
    case TypeDescription::SFCZVariable:
	translation=ZFaceBased;
	break;
    case TypeDescription::ParticleVariable:
	translation=CellBased;
	break;
    default:
	throw InternalError("Unknown variable type in Patch::getVariableExtents (from TypeDescription::Type)");
    }
    computeVariableExtents(translation, gtype, numGhostCells,
			   neighbors, low, high);
}

#if 0
// numGC = number of ghost cells
IntVector Patch::getGhostCellLowIndex(const int numGC) const
{  IntVector h(d_lowIndex-
	       IntVector(getBCType(xminus) == Neighbor?numGC:0,
			 getBCType(yminus) == Neighbor?numGC:0,
			 getBCType(zminus) == Neighbor?numGC:0));
   return h;
}

IntVector Patch::getGhostCellHighIndex(const int numGC) const
{  IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?numGC:0,
			 getBCType(yplus) == Neighbor?numGC:0,
			 getBCType(zplus) == Neighbor?numGC:0));
   return h;
}

// numGC = number of ghost cells
IntVector Patch::getGhostSFCXLowIndex(const int numGC) const
{  IntVector h(d_lowIndex-
	       IntVector(getBCType(xminus) == Neighbor?numGC:0,
			 getBCType(yminus) == Neighbor?numGC:0,
			 getBCType(zminus) == Neighbor?numGC:0));
   return h;
}

IntVector Patch::getGhostSFCXHighIndex(const int numGC) const
{  IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?numGC:1,
			 getBCType(yplus) == Neighbor?numGC:0,
			 getBCType(zplus) == Neighbor?numGC:0));
   return h;
}

// numGC = number of ghost cells
IntVector Patch::getGhostSFCYLowIndex(const int numGC) const
{  IntVector h(d_lowIndex-
	       IntVector(getBCType(xminus) == Neighbor?numGC:0,
			 getBCType(yminus) == Neighbor?numGC:0,
			 getBCType(zminus) == Neighbor?numGC:0));
   return h;
}

IntVector Patch::getGhostSFCYHighIndex(const int numGC) const
{  IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?numGC:0,
			 getBCType(yplus) == Neighbor?numGC:1,
			 getBCType(zplus) == Neighbor?numGC:0));
   return h;
}

// numGC = number of ghost cells
IntVector Patch::getGhostSFCZLowIndex(const int numGC) const
{  IntVector h(d_lowIndex-
	       IntVector(getBCType(xminus) == Neighbor?numGC:0,
			 getBCType(yminus) == Neighbor?numGC:0,
			 getBCType(zminus) == Neighbor?numGC:0));
   return h;
}

IntVector Patch::getGhostSFCZHighIndex(const int numGC) const
{  IntVector h(d_highIndex+
	       IntVector(getBCType(xplus) == Neighbor?numGC:0,
			 getBCType(yplus) == Neighbor?numGC:0,
			 getBCType(zplus) == Neighbor?numGC:1));
   return h;
}
#endif

//
// $Log$
// Revision 1.21  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.20  2000/09/22 22:06:16  rawat
// fixed some bugs in staggered variables call
//
// Revision 1.19  2000/08/23 22:32:07  dav
// changed output operator to use a reference, and not a pointer to a patch
//
// Revision 1.18  2000/08/02 03:29:33  jas
// Fixed grid bcs problem.
//
// Revision 1.17  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.16  2000/07/11 15:21:24  kuzimmer
// Patch::getCellIterator()
//
// Revision 1.15  2000/06/27 23:18:17  rawat
// implemented Staggered cell variables. Modified Patch.cc to get ghostcell
// and staggered cell indexes.
//
// Revision 1.14  2000/06/27 22:49:04  jas
// Added grid boundary condition support.
//
// Revision 1.13  2000/06/26 17:09:01  bigler
// Added getNodeIterator which takes a Box and returns the iterator
// that will loop over the nodes that lie withing the Box.
//
// Revision 1.12  2000/06/16 05:19:21  sparker
// Changed arrays to fortran order
//
// Revision 1.11  2000/06/15 21:57:19  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/06/14 19:58:03  guilkey
// Added a different version of findCell.
//
// Revision 1.9  2000/06/13 21:28:30  jas
// Added missing TypeUtils.h for fun_forgottherestofname and copy constructor
// was wrong for CellIterator.
//
// Revision 1.8  2000/06/08 17:47:47  dav
// longer error message
//
// Revision 1.7  2000/06/07 18:31:00  tan
// Requirement for getHighGhostCellIndex() and getLowGhostCellIndex()
// cancelled.
//
// Revision 1.6  2000/06/05 19:25:14  tan
// I need the following two functions,
// (1) IntVector getHighGhostCellIndex() const;
// (2) IntVector getLowGhostCellIndex() const;
// The temporary empty functions are created.
//
// Revision 1.5  2000/06/04 04:36:07  tan
// Added function findNodesFromCell() to find the 8 neighboring node indexes
// according to a given cell index.
//
// Revision 1.4  2000/06/02 20:44:56  tan
// Corrected a mistake in function findCellsFromNode().
//
// Revision 1.3  2000/06/02 19:58:01  tan
// Added function findCellsFromNode() to find the 8 neighboring cell
// indexes according to a given node index.
//
// Revision 1.2  2000/06/01 22:14:06  tan
// Added findCell(const Point& pos).
//
// Revision 1.1  2000/05/30 20:19:31  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.20  2000/05/28 17:25:06  dav
// adding mpi stuff
//
// Revision 1.19  2000/05/20 08:09:26  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.18  2000/05/15 19:39:49  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.17  2000/05/10 20:03:02  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.16  2000/05/09 03:24:39  jas
// Added some enums for grid boundary conditions.
//
// Revision 1.15  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.14  2000/05/05 06:42:45  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.13  2000/05/04 19:06:48  guilkey
// Added the beginnings of grid boundary conditions.  Functions still
// need to be filled in.
//
// Revision 1.12  2000/05/02 20:30:59  jas
// Fixed the findCellAndShapeDerivatives.
//
// Revision 1.11  2000/05/02 20:13:05  sparker
// Implemented findCellAndWeights
//
// Revision 1.10  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.9  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.8  2000/04/28 03:58:20  sparker
// Fixed countParticles
// Implemented createParticles, which doesn't quite work yet because the
//   data warehouse isn't there yet.
// Reduced the number of particles in the bar problem so that it will run
//   quickly during development cycles
//
// Revision 1.7  2000/04/27 23:18:50  sparker
// Added problem initialization for MPM
//
// Revision 1.6  2000/04/26 06:48:54  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/12 23:00:49  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

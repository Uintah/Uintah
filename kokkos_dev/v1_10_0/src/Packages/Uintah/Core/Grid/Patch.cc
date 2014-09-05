
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Packages/Uintah/Core/Math/Primes.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Core/Containers/StaticArray.h>

#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
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

    if(patches.find(d_id) != patches.end()){
      cerr << "id=" << d_id << '\n';
      SCI_THROW(InternalError("duplicate patch!"));
    }
    patches[d_id]=this;
    in_database=true;
  } else {
    in_database=false;
    if(d_id >= *ids)
      ids->set(d_id+1);
   }

   d_bcs.resize(numFaces);

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
      d_bcs(realPatch->d_bcs),
      have_layout(realPatch->have_layout),
      layouthint(realPatch->layouthint)
{
  // make the id be -1000 * realPatch id - some first come, first serve index
  if(!ids){
    d_id = -1000 * realPatch->d_id; // temporary
   ids_init.lock();    
    int index = 1;
    while (patches.find(d_id - index) != patches.end()){
      if (++index >= 27) {
	SCI_THROW(InternalError("A real patch shouldn't have more than 26 (3*3*3 - 1) virtual patches"));
      }
    }
    d_id -= index;
    ASSERT(patches.find(d_id) == patches.end());    
    patches[d_id]=this;
    in_database = true;
   ids_init.unlock();    
  }      
  
  for (int i = 0; i < numFaces; i++)
    d_bctypes[i] = realPatch->d_bctypes[i];
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
                               IntVector ni[8],
                               double S[8]) const
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

#if 0
void Patch::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                  IntVector ni[8],
                                                  double S[8],
                                                  Vector d_S[8],
						  int& n) const
{
   Point cellpos = d_level->positionToIndex(pos);
   int ix = Floor(cellpos.x());
   int iy = Floor(cellpos.y());
   int iz = Floor(cellpos.z());
   double fx = cellpos.x() - ix;
   double fy = cellpos.y() - iy;
   double fz = cellpos.z() - iz;
   double fx1 = 1-fx;
   double fy1 = 1-fy;
   double fz1 = 1-fz;
   if(??? ){
   ni[0] = IntVector(ix, iy, iz);
   ni[1] = IntVector(ix, iy, iz+1);
   ni[2] = IntVector(ix, iy+1, iz);
   ni[3] = IntVector(ix, iy+1, iz+1);
   ni[4] = IntVector(ix+1, iy, iz);
   ni[5] = IntVector(ix+1, iy, iz+1);
   ni[6] = IntVector(ix+1, iy+1, iz);
   ni[7] = IntVector(ix+1, iy+1, iz+1);
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
#endif


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

void Patch::findCellAndWeights27(const Point& pos,
                                 IntVector ni[27],
                                 double S[27], const Vector& size) const
{
   Point cellpos = d_level->positionToIndex(pos);
   int ix = Floor(cellpos.x());
   int iy = Floor(cellpos.y());
   int iz = Floor(cellpos.z());
   int nnx,nny,nnz;
   double lx = size.x()/2.;
   double ly = size.y()/2.;
   double lz = size.z()/2.;

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

   // (x_p - x_v)/L
   double px0 = cellpos.x() - (ix);
   double px1 = cellpos.x() - (ix+1);
   double px2 = cellpos.x() - (ix + nnx);
   double py0 = cellpos.y() - (iy);
   double py1 = cellpos.y() - (iy+1);
   double py2 = cellpos.y() - (iy + nny);
   double pz0 = cellpos.z() - (iz);
   double pz1 = cellpos.z() - (iz+1);
   double pz2 = cellpos.z() - (iz + nnz);
   double fx[3], fy[3], fz[3];

   if(px0 <= lx){
     fx[0] = 1. - (px0*px0 + (lx)*(lx))/(2*lx);
     fx[1] = (1. + lx + px1)*(1. + lx + px1)/(4*lx);
     fx[2] = (1. + lx - px2)*(1. + lx - px2)/(4*lx);
   }
   else if(px0 > lx && px0 <= (1.-lx)){
     fx[0] = 1. - px0;
     fx[1] = 1. + px1;
     fx[2] = 0.;
   }
   else if(px0 > (1.-lx)){
     fx[0] = (1. + lx - px0)*(1. + lx - px0)/(4*lx);
     fx[1] = 1. - (px1*px1 + (lx)*(lx))/(2*lx);
     fx[2] = (1. + lx + px2)*(1. + lx + px2)/(4*lx);
   }

   if(py0 <= ly){
     fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2*ly);
     fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4*ly);
     fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4*ly);
   }
   else if(py0 > ly && py0 <= (1.-ly)){
     fy[0] = 1. - py0;
     fy[1] = 1. + py1;
     fy[2] = 0.;
   }
   else if(py0 > (1.-ly)){
     fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4*ly);
     fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2*ly);
     fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4*ly);
   }

   if(pz0 <= lz){
     fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
     fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4*lz);
     fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4*lz);
   }
   else if(pz0 > lz && pz0 <= (1.-lz)){
     fz[0] = 1. - pz0;
     fz[1] = 1. + pz1;
     fz[2] = 0.;
   }
   else if(pz0 > (1.-lz)){
     fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4*lz);
     fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2*lz);
     fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4*lz);
   }

   S[0]  = fx[0]*fy[0]*fz[0];
   S[1]  = fx[1]*fy[0]*fz[0];
   S[2]  = fx[2]*fy[0]*fz[0];
   S[3]  = fx[0]*fy[1]*fz[0];
   S[4]  = fx[1]*fy[1]*fz[0];
   S[5]  = fx[2]*fy[1]*fz[0];
   S[6]  = fx[0]*fy[2]*fz[0];
   S[7]  = fx[1]*fy[2]*fz[0];
   S[8]  = fx[2]*fy[2]*fz[0];
   S[9]  = fx[0]*fy[0]*fz[1];
   S[10] = fx[1]*fy[0]*fz[1];
   S[11] = fx[2]*fy[0]*fz[1];
   S[12] = fx[0]*fy[1]*fz[1];
   S[13] = fx[1]*fy[1]*fz[1];
   S[14] = fx[2]*fy[1]*fz[1];
   S[15] = fx[0]*fy[2]*fz[1];
   S[16] = fx[1]*fy[2]*fz[1];
   S[17] = fx[2]*fy[2]*fz[1];
   S[18] = fx[0]*fy[0]*fz[2];
   S[19] = fx[1]*fy[0]*fz[2];
   S[20] = fx[2]*fy[0]*fz[2];
   S[21] = fx[0]*fy[1]*fz[2];
   S[22] = fx[1]*fy[1]*fz[2];
   S[23] = fx[2]*fy[1]*fz[2];
   S[24] = fx[0]*fy[2]*fz[2];
   S[25] = fx[1]*fy[2]*fz[2];
   S[26] = fx[2]*fy[2]*fz[2];

}

void Patch::findCellAndShapeDerivatives27(const Point& pos,
                                          IntVector ni[27],
                                          Vector d_S[27],
                                          const Vector& size) const
{
   Point cellpos = d_level->positionToIndex(pos);
   int ix = Floor(cellpos.x());
   int iy = Floor(cellpos.y());
   int iz = Floor(cellpos.z());
   int nnx,nny,nnz;
   double lx = size.x()/2.;
   double ly = size.y()/2.;
   double lz = size.z()/2.;

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

   // (x_p - x_v)/L
   double px0 = cellpos.x() - (ix);
   double px1 = cellpos.x() - (ix+1);
   double px2 = cellpos.x() - (ix + nnx);
   double py0 = cellpos.y() - (iy);
   double py1 = cellpos.y() - (iy+1);
   double py2 = cellpos.y() - (iy + nny);
   double pz0 = cellpos.z() - (iz);
   double pz1 = cellpos.z() - (iz+1);
   double pz2 = cellpos.z() - (iz + nnz);
   double fx[3], fy[3], fz[3], dfx[3], dfy[3], dfz[3];

   if(px0 <= lx){
     fx[0]  = 1. - (px0*px0 + (lx)*(lx))/(2.*lx);
     fx[1]  = (1. + lx + px1)*(1. + lx + px1)/(4.*lx);
     fx[2]  = (1. + lx - px2)*(1. + lx - px2)/(4.*lx);
     dfx[0] = -px0/lx;
     dfx[1] =  (1. + lx + px1)/(2.*lx);
     dfx[2] = -(1. + lx - px2)/(2.*lx);
   }
   else if(px0 > lx && px0 <= (1-lx)){
     fx[0]  = 1. - px0;
     fx[1]  = 1. + px1;
     fx[2]  = 0.;
     dfx[0] = -1.;
     dfx[1] =  1.;
     dfx[2] =  0.;
   }
   else if(px0 > (1-lx)){
     fx[0]  = (1. + lx - px0)*(1. + lx - px0)/(4.*lx);
     fx[1]  = 1. - (px1*px1 + (lx)*(lx))/(2.*lx);
     fx[2]  = (1. + lx + px2)*(1. + lx + px2)/(4.*lx);
     dfx[0] = -(1. + lx - px0)/(2.*lx);
     dfx[1] = -px1/lx;
     dfx[2] = (1. + lx + px2)/(2.*lx);
   }

   if(py0 <= ly){
     fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2.*ly);
     fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4.*ly);
     fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4.*ly);
     dfy[0] = -py0/ly;
     dfy[1] =  (1. + ly + py1)/(2.*ly);
     dfy[2] = -(1. + ly - py2)/(2.*ly);
   }
   else if(py0 > ly && py0 <= (1-ly)){
     fy[0] = 1. - py0;
     fy[1] = 1. + py1;
     fy[2] = 0.;
     dfy[0] = -1.;
     dfy[1] =  1.;
     dfy[2] =  0.;
   }
   else if(py0 > (1-ly)){
     fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4.*ly);
     fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2.*ly);
     fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4.*ly);
     dfy[0] = -(1. + ly - py0)/(2.*ly);
     dfy[1] = -py1/ly;
     dfy[2] = (1. + ly + py2)/(2.*ly);
   }

   if(pz0 <= lz){
     fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
     fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4.*lz);
     fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4.*lz);
     dfz[0] = -pz0/lz;
     dfz[1] =  (1. + lz + pz1)/(2.*lz);
     dfz[2] = -(1. + lz - pz2)/(2.*lz);
   }
   else if(pz0 > lz && pz0 <= (1-lz)){
     fz[0] = 1. - pz0;
     fz[1] = 1. + pz1;
     fz[2] = 0.;
     dfz[0] = -1.;
     dfz[1] =  1.;
     dfz[2] =  0.;
   }
   else if(pz0 > (1-lz)){
     fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4.*lz);
     fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2.*lz);
     fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4.*lz);
     dfz[0] = -(1. + lz - pz0)/(2.*lz);
     dfz[1] = -pz1/lz;
     dfz[2] = (1. + lz + pz2)/(2.*lz);
   }

   d_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
   d_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
   d_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
   d_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
   d_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
   d_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
   d_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
   d_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
   d_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);

   d_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
   d_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
   d_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
   d_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
   d_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
   d_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
   d_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
   d_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
   d_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);

   d_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
   d_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
   d_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
   d_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
   d_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
   d_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
   d_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
   d_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
   d_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

}

void Patch::findCellAndWeightsAndShapeDerivatives27(const Point& pos,
                                                    IntVector ni[27],
                                                    double S[27],
                                                    Vector d_S[27],
                                                    const Vector& size) const
{
   Point cellpos = d_level->positionToIndex(pos);
   int ix = Floor(cellpos.x());
   int iy = Floor(cellpos.y());
   int iz = Floor(cellpos.z());
   int nnx,nny,nnz;
   double lx = size.x()/2.;
   double ly = size.y()/2.;
   double lz = size.z()/2.;

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

   // (x_p - x_v)/L
   double px0 = cellpos.x() - (ix);
   double px1 = cellpos.x() - (ix+1);
   double px2 = cellpos.x() - (ix + nnx);
   double py0 = cellpos.y() - (iy);
   double py1 = cellpos.y() - (iy+1);
   double py2 = cellpos.y() - (iy + nny);
   double pz0 = cellpos.z() - (iz);
   double pz1 = cellpos.z() - (iz+1);
   double pz2 = cellpos.z() - (iz + nnz);
   double fx[3], fy[3], fz[3], dfx[3], dfy[3], dfz[3];

   if(px0 <= lx){
     fx[0]  = 1. - (px0*px0 + (lx)*(lx))/(2.*lx);
     fx[1]  = (1. + lx + px1)*(1. + lx + px1)/(4.*lx);
     fx[2]  = (1. + lx - px2)*(1. + lx - px2)/(4.*lx);
     dfx[0] = -px0/lx;
     dfx[1] =  (1. + lx + px1)/(2.*lx);
     dfx[2] = -(1. + lx - px2)/(2.*lx);
   }
   else if(px0 > lx && px0 <= (1-lx)){
     fx[0]  = 1. - px0;
     fx[1]  = 1. + px1;
     fx[2]  = 0.;
     dfx[0] = -1.;
     dfx[1] =  1.;
     dfx[2] =  0.;
   }
   else if(px0 > (1-lx)){
     fx[0]  = (1. + lx - px0)*(1. + lx - px0)/(4.*lx);
     fx[1]  = 1. - (px1*px1 + (lx)*(lx))/(2.*lx);
     fx[2]  = (1. + lx + px2)*(1. + lx + px2)/(4.*lx);
     dfx[0] = -(1. + lx - px0)/(2.*lx);
     dfx[1] = -px1/lx;
     dfx[2] = (1. + lx + px2)/(2.*lx);
   }

   if(py0 <= ly){
     fy[0] = 1. - (py0*py0 + (ly)*(ly))/(2.*ly);
     fy[1] = (1. + ly + py1)*(1. + ly + py1)/(4.*ly);
     fy[2] = (1. + ly - py2)*(1. + ly - py2)/(4.*ly);
     dfy[0] = -py0/ly;
     dfy[1] =  (1. + ly + py1)/(2.*ly);
     dfy[2] = -(1. + ly - py2)/(2.*ly);
   }
   else if(py0 > ly && py0 <= (1-ly)){
     fy[0] = 1. - py0;
     fy[1] = 1. + py1;
     fy[2] = 0.;
     dfy[0] = -1.;
     dfy[1] =  1.;
     dfy[2] =  0.;
   }
   else if(py0 > (1-ly)){
     fy[0] = (1. + ly - py0)*(1. + ly - py0)/(4.*ly);
     fy[1] = 1. - (py1*py1 + (ly)*(ly))/(2.*ly);
     fy[2] = (1. + ly + py2)*(1. + ly + py2)/(4.*ly);
     dfy[0] = -(1. + ly - py0)/(2.*ly);
     dfy[1] = -py1/ly;
     dfy[2] = (1. + ly + py2)/(2.*ly);
   }

   if(pz0 <= lz){
     fz[0] = 1. - (pz0*pz0 + (lz)*(lz))/(2*lz);
     fz[1] = (1. + lz + pz1)*(1. + lz + pz1)/(4.*lz);
     fz[2] = (1. + lz - pz2)*(1. + lz - pz2)/(4.*lz);
     dfz[0] = -pz0/lz;
     dfz[1] =  (1. + lz + pz1)/(2.*lz);
     dfz[2] = -(1. + lz - pz2)/(2.*lz);
   }
   else if(pz0 > lz && pz0 <= (1-lz)){
     fz[0] = 1. - pz0;
     fz[1] = 1. + pz1;
     fz[2] = 0.;
     dfz[0] = -1.;
     dfz[1] =  1.;
     dfz[2] =  0.;
   }
   else if(pz0 > (1-lz)){
     fz[0] = (1. + lz - pz0)*(1. + lz - pz0)/(4.*lz);
     fz[1] = 1. - (pz1*pz1 + (lz)*(lz))/(2.*lz);
     fz[2] = (1. + lz + pz2)*(1. + lz + pz2)/(4.*lz);
     dfz[0] = -(1. + lz - pz0)/(2.*lz);
     dfz[1] = -pz1/lz;
     dfz[2] = (1. + lz + pz2)/(2.*lz);
   }

   S[0]  = fx[0]*fy[0]*fz[0];
   S[1]  = fx[1]*fy[0]*fz[0];
   S[2]  = fx[2]*fy[0]*fz[0];
   S[3]  = fx[0]*fy[1]*fz[0];
   S[4]  = fx[1]*fy[1]*fz[0];
   S[5]  = fx[2]*fy[1]*fz[0];
   S[6]  = fx[0]*fy[2]*fz[0];
   S[7]  = fx[1]*fy[2]*fz[0];
   S[8]  = fx[2]*fy[2]*fz[0];
   S[9]  = fx[0]*fy[0]*fz[1];
   S[10] = fx[1]*fy[0]*fz[1];
   S[11] = fx[2]*fy[0]*fz[1];
   S[12] = fx[0]*fy[1]*fz[1];
   S[13] = fx[1]*fy[1]*fz[1];
   S[14] = fx[2]*fy[1]*fz[1];
   S[15] = fx[0]*fy[2]*fz[1];
   S[16] = fx[1]*fy[2]*fz[1];
   S[17] = fx[2]*fy[2]*fz[1];
   S[18] = fx[0]*fy[0]*fz[2];
   S[19] = fx[1]*fy[0]*fz[2];
   S[20] = fx[2]*fy[0]*fz[2];
   S[21] = fx[0]*fy[1]*fz[2];
   S[22] = fx[1]*fy[1]*fz[2];
   S[23] = fx[2]*fy[1]*fz[2];
   S[24] = fx[0]*fy[2]*fz[2];
   S[25] = fx[1]*fy[2]*fz[2];
   S[26] = fx[2]*fy[2]*fz[2];

   d_S[0]  = Vector(dfx[0]*fy[0]*fz[0],fx[0]*dfy[0]*fz[0],fx[0]*fy[0]*dfz[0]);
   d_S[1]  = Vector(dfx[1]*fy[0]*fz[0],fx[1]*dfy[0]*fz[0],fx[1]*fy[0]*dfz[0]);
   d_S[2]  = Vector(dfx[2]*fy[0]*fz[0],fx[2]*dfy[0]*fz[0],fx[2]*fy[0]*dfz[0]);
   d_S[3]  = Vector(dfx[0]*fy[1]*fz[0],fx[0]*dfy[1]*fz[0],fx[0]*fy[1]*dfz[0]);
   d_S[4]  = Vector(dfx[1]*fy[1]*fz[0],fx[1]*dfy[1]*fz[0],fx[1]*fy[1]*dfz[0]);
   d_S[5]  = Vector(dfx[2]*fy[1]*fz[0],fx[2]*dfy[1]*fz[0],fx[2]*fy[1]*dfz[0]);
   d_S[6]  = Vector(dfx[0]*fy[2]*fz[0],fx[0]*dfy[2]*fz[0],fx[0]*fy[2]*dfz[0]);
   d_S[7]  = Vector(dfx[1]*fy[2]*fz[0],fx[1]*dfy[2]*fz[0],fx[1]*fy[2]*dfz[0]);
   d_S[8]  = Vector(dfx[2]*fy[2]*fz[0],fx[2]*dfy[2]*fz[0],fx[2]*fy[2]*dfz[0]);

   d_S[9]  = Vector(dfx[0]*fy[0]*fz[1],fx[0]*dfy[0]*fz[1],fx[0]*fy[0]*dfz[1]);
   d_S[10] = Vector(dfx[1]*fy[0]*fz[1],fx[1]*dfy[0]*fz[1],fx[1]*fy[0]*dfz[1]);
   d_S[11] = Vector(dfx[2]*fy[0]*fz[1],fx[2]*dfy[0]*fz[1],fx[2]*fy[0]*dfz[1]);
   d_S[12] = Vector(dfx[0]*fy[1]*fz[1],fx[0]*dfy[1]*fz[1],fx[0]*fy[1]*dfz[1]);
   d_S[13] = Vector(dfx[1]*fy[1]*fz[1],fx[1]*dfy[1]*fz[1],fx[1]*fy[1]*dfz[1]);
   d_S[14] = Vector(dfx[2]*fy[1]*fz[1],fx[2]*dfy[1]*fz[1],fx[2]*fy[1]*dfz[1]);
   d_S[15] = Vector(dfx[0]*fy[2]*fz[1],fx[0]*dfy[2]*fz[1],fx[0]*fy[2]*dfz[1]);
   d_S[16] = Vector(dfx[1]*fy[2]*fz[1],fx[1]*dfy[2]*fz[1],fx[1]*fy[2]*dfz[1]);
   d_S[17] = Vector(dfx[2]*fy[2]*fz[1],fx[2]*dfy[2]*fz[1],fx[2]*fy[2]*dfz[1]);

   d_S[18] = Vector(dfx[0]*fy[0]*fz[2],fx[0]*dfy[0]*fz[2],fx[0]*fy[0]*dfz[2]);
   d_S[19] = Vector(dfx[1]*fy[0]*fz[2],fx[1]*dfy[0]*fz[2],fx[1]*fy[0]*dfz[2]);
   d_S[20] = Vector(dfx[2]*fy[0]*fz[2],fx[2]*dfy[0]*fz[2],fx[2]*fy[0]*dfz[2]);
   d_S[21] = Vector(dfx[0]*fy[1]*fz[2],fx[0]*dfy[1]*fz[2],fx[0]*fy[1]*dfz[2]);
   d_S[22] = Vector(dfx[1]*fy[1]*fz[2],fx[1]*dfy[1]*fz[2],fx[1]*fy[1]*dfz[2]);
   d_S[23] = Vector(dfx[2]*fy[1]*fz[2],fx[2]*dfy[1]*fz[2],fx[2]*fy[1]*dfz[2]);
   d_S[24] = Vector(dfx[0]*fy[2]*fz[2],fx[0]*dfy[2]*fz[2],fx[0]*fy[2]*dfz[2]);
   d_S[25] = Vector(dfx[1]*fy[2]*fz[2],fx[1]*dfy[2]*fz[2],fx[1]*fy[2]*dfz[2]);
   d_S[26] = Vector(dfx[2]*fy[2]*fz[2],fx[2]*dfy[2]*fz[2],fx[2]*fy[2]*dfz[2]);

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

const BoundCondBase*
Patch::getBCValues(int mat_id,string type,Patch::FaceType face) const
{
  return d_bcs[face].getBCValues(mat_id,type);
}


void
Patch::getFace(FaceType face, const IntVector& insideOffset,
	       const IntVector& outsideOffset,
	       IntVector& l, IntVector& h) const
{
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

void
Patch::getFaceNodes(FaceType face, int offset,IntVector& l, IntVector& h) const
{
   l=getNodeLowIndex();
   h=getNodeHighIndex();
   switch(face){
   case xminus:
      l.x(l.x()-offset);
      h.x(l.x()+1-offset);
      break;
   case xplus:
      l.x(h.x()-1+offset);
      h.x(h.x()+offset);
      break;
   case yminus:
      l.y(l.y()-offset);
      h.y(l.y()+1-offset);
      break;
   case yplus:
      l.y(h.y()-1+offset);
      h.y(h.y()+offset);
      break;
   case zminus:
      l.z(l.z()-offset);
      h.z(l.z()+1-offset);
      break;
   case zplus:
      l.z(h.z()-1+offset);
      h.z(h.z()+offset);
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
// if domain == plusEdgeCells this includes the edge and corner cells.
CellIterator    
Patch::getFaceCellIterator(const FaceType& face, const string& domain) const
{ 
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
				   Level::selectType& neighbors,
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
				   Level::selectType& neighbors,
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
				 Level::selectType& patches) const
{
  const LevelP& otherLevel = d_level->getRelativeLevel(levelOffset);
  IntVector low = 
    otherLevel->getCellIndex(d_level->getCellPosition(getLowIndex()));
  IntVector high =
    otherLevel->getCellIndex(d_level->getCellPosition(getHighIndex()));
  otherLevel->selectPatches(low, high, patches); 
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


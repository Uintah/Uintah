#include "BrokenCellShapeFunction.h"

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Lattice.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/ParticlesNeighbor.h>

namespace Uintah {
BrokenCellShapeFunction::
BrokenCellShapeFunction( const Lattice& lattice,
                         const ParticleVariable<int>& pIsBroken,
			 const ParticleVariable<Vector>& pCrackSurfaceNormal,
			 const ParticleVariable<double>& pMicrocrackSize,
			 const ParticleVariable<double>& pMicrocracPosition )


: d_lattice(lattice),
  d_pIsBroken(pIsBroken),
  d_pCrackSurfaceNormal(pCrackSurfaceNormal),
  d_pMicrocrackSize(pMicrocrackSize),
  d_pMicrocracPosition(pMicrocracPosition)
{
}

bool
BrokenCellShapeFunction::
findCellAndWeights( int partIdx, 
                    IntVector nodeIdx[8], 
                    bool visiable[8],
                    double S[8] ) const
{
  d_lattice.getPatch()->findCellAndWeights(d_lattice.getpX()[partIdx], 
     nodeIdx, S);

  for(int i=0;i<8;++i) {
    visiable[i] = getVisibility( partIdx,nodeIdx[i] );
  }
  return true;
}

bool
BrokenCellShapeFunction::
findCellAndShapeDerivatives( int partIdx, 
                             IntVector nodeIdx[8], 
                             bool visiable[8],
                             Vector d_S[8] ) const
{
  d_lattice.getPatch()->findCellAndShapeDerivatives(d_lattice.getpX()[partIdx], 
     nodeIdx, d_S);
  
  for(int i=0;i<8;++i) {
    visiable[i] = getVisibility( partIdx,nodeIdx[i] );
  }

  return true;
}

bool
BrokenCellShapeFunction::
findCellAndWeightsAndShapeDerivatives( int partIdx, 
                             IntVector nodeIdx[8], 
                             bool visiable[8],
			     double S[8],
                             Vector d_S[8] ) const
{
  d_lattice.getPatch()->findCellAndWeights(d_lattice.getpX()[partIdx], 
     nodeIdx, S);

  d_lattice.getPatch()->findCellAndShapeDerivatives(d_lattice.getpX()[partIdx], 
     nodeIdx, d_S);
  
  for(int i=0;i<8;++i) {
    visiable[i] = getVisibility( partIdx,nodeIdx[i] );
  }

  return true;
}

bool
BrokenCellShapeFunction::
getVisibility(int partIdx,const IntVector& nodeIdx) const
{
  IntVector cellIdx;
  d_lattice.getPatch()->findCell(d_lattice.getpX()[partIdx],cellIdx);

  ParticlesNeighbor particles( d_lattice.getpX(),
			       d_pIsBroken,
			       d_pCrackSurfaceNormal,
			       d_pMicrocrackSize,
			       d_pMicrocracPosition);
  particles.buildIn(cellIdx,d_lattice);

  return particles.visible(d_lattice.getpX()[partIdx],
                           d_lattice.getPatch()->nodePosition(nodeIdx) );
}
} // End namespace Uintah



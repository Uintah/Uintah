#include "BrokenCellShapeFunction.h"

#include <Uintah/Grid/Patch.h>
#include <Uintah/Components/MPM/Fracture/Lattice.h>

namespace Uintah {
namespace MPM {

BrokenCellShapeFunction::
BrokenCellShapeFunction( const Lattice& lattice,
                         const ParticleVariable<int>& pIsBroken,
			 const ParticleVariable<Vector>& pCrackSurfaceNormal )


: d_lattice(lattice),
  d_pIsBroken(pIsBroken),
  d_pCrackSurfaceNormal(pCrackSurfaceNormal)
{
}

bool
BrokenCellShapeFunction::
findCellAndWeights( int partIdx, 
                    IntVector nodeIdx[8], 
                    bool visiable[8],
                    double S[8] ) const
{
  double completeShape[8];
  
  if( !d_lattice.getPatch()->findCellAndWeights(d_lattice.getpX()[partIdx], 
     nodeIdx, completeShape) )
  return false;

  for(int i=0;i<8;++i) {
    visiable[i] = getVisiability( partIdx,nodeIdx[i] );
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
  double completeShape[8];
  if( !d_lattice.getPatch()->findCellAndWeights(d_lattice.getpX()[partIdx], 
     nodeIdx, completeShape) )
  return false;

  double completeShapeDerivative[8];
  if( !d_lattice.getPatch()->findCellAndWeights(d_lattice.getpX()[partIdx], 
     nodeIdx, completeShapeDerivative) )
  return false;
  
  for(int i=0;i<8;++i) {
    visiable[i] = getVisiability( partIdx,nodeIdx[i] );
  }

  return true;
}

bool
BrokenCellShapeFunction::
getVisiability(int partIdx,const IntVector& nodeIdx) const
{
  return true;
}

} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.3  2000/09/05 06:59:28  tan
// Applied BrokenCellShapeFunction to SerialMPM::interpolateToParticlesAndUpdate
// where fracture is involved.
//
// Revision 1.2  2000/09/05 06:34:54  tan
// Introduced BrokenCellShapeFunction for SerialMPM::interpolateParticlesToGrid
// where farcture is involved.
//
// Revision 1.1  2000/08/11 03:13:42  tan
// Created BrokenCellShapeFunction to handle Shape functions (including Derivatives)
// for a cell containing cracked particles.
//

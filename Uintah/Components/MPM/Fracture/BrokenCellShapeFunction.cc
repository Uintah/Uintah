#include "BrokenCellShapeFunction.h"

#include <Uintah/Grid/Patch.h>

namespace Uintah {
namespace MPM {

BrokenCellShapeFunction::
BrokenCellShapeFunction( const Patch& patch,
                         const Lattice& lattice,
                         const ParticleVariable<Point>& pX,
                         const ParticleVariable<Vector>& pCrackSurfaceNormal )
: d_patch(patch),
  d_lattice(lattice),
  d_pX(pX),
  d_pCrackSurfaceNormal(pCrackSurfaceNormal)
{
}

void
BrokenCellShapeFunction::
findCellAndWeights( int partIdx, 
                    IntVector nodeIdx[8], 
                    bool visiable[8],
                    double S[8] ) const
{
  double completeShape[8];
  
  if( !d_patch.findCellAndWeights(d_pX[partIdx], nodeIdx, completeShape) )
  {
    throw InternalError("Particle not in patch");
  }

  for(int i=0;i<8;++i) {
    visiable[i] = getVisiability( partIdx,nodeIdx[i] );
  }
}

void
BrokenCellShapeFunction::
findCellAndShapeDerivatives( int partIdx, 
                             IntVector nodeIdx[8], 
                             bool visiable[8],
                             double d_S[8][3] ) const
{
  double completeShapeDerivative[8];
  if( !d_patch.findCellAndWeights(d_pX[partIdx], nodeIdx, completeShapeDerivative) )
  {
    throw InternalError("Particle not in patch");
  }
  
  for(int i=0;i<8;++i) {
    visiable[i] = getVisiability( partIdx,nodeIdx[i] );
  }
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
// Revision 1.1  2000/08/11 03:13:42  tan
// Created BrokenCellShapeFunction to handle Shape functions (including Derivatives)
// for a cell containing cracked particles.
//

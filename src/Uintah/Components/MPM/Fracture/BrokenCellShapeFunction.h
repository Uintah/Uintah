#ifndef __Uintah_MPM_BrokenCellShapeFunction__
#define __Uintah_MPM_BrokenCellShapeFunction__

#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Geometry/Point.h>

class Matrix3;

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Point;
class Lattice;

class BrokenCellShapeFunction {
public:

        BrokenCellShapeFunction( const Patch& patch,
                                 const Lattice& lattice,
                                 const ParticleVariable<Point>& pX,
                                 const ParticleVariable<Vector>& pCrackSurfaceNormal );

  void  findCellAndWeights( int partIdx, 
                            IntVector nodeIdx[8], 
                            bool visiable[8],
                            double S[8] ) const;

  void  findCellAndShapeDerivatives( int partIdx, 
                                     IntVector nodeIdx[8],
                                     bool visiable[8],
                                     double d_S[8][3] ) const;

  bool  getVisiability(int partIdx,const IntVector& nodeIdx) const;

private:
  const ParticleVariable<Point>&  d_pX;
  const ParticleVariable<Vector>& d_pCrackSurfaceNormal;
  const Patch&                    d_patch;
  const Lattice&                  d_lattice;
};

} //namespace MPM
} //namespace Uintah

#endif //__Uintah_MPM_BrokenCellShapeFunction__

// $Log$
// Revision 1.1  2000/08/11 03:13:30  tan
// Created BrokenCellShapeFunction to handle Shape functions (including Derivatives)
// for a cell containing cracked particles.
//

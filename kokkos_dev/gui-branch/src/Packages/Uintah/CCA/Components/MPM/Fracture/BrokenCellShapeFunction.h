#ifndef __Uintah_MPM_BrokenCellShapeFunction__
#define __Uintah_MPM_BrokenCellShapeFunction__

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Core/Geometry/Point.h>

class Matrix3;

namespace Uintah {
using namespace SCIRun;
class Lattice;

class BrokenCellShapeFunction {
public:

        BrokenCellShapeFunction( const Lattice& lattice,
                                 const ParticleVariable<int>& pIsBroken,
				 const ParticleVariable<Vector>& pCrackSurfaceNormal,
				 const ParticleVariable<double>& pMicrocrackSize,
			         const ParticleVariable<double>& pMicrocracPosition );

  bool  findCellAndWeights( int partIdx, 
                            IntVector nodeIdx[8], 
                            bool visiable[8],
                            double S[8] ) const;

  bool  findCellAndShapeDerivatives( int partIdx, 
                             IntVector nodeIdx[8], 
                             bool visiable[8],
                             Vector d_S[8] ) const;

  bool  findCellAndWeightsAndShapeDerivatives( int partIdx, 
                             IntVector nodeIdx[8], 
                             bool visiable[8],
			     double S[8],
                             Vector d_S[8] ) const;

  bool  getVisibility(int partIdx,const IntVector& nodeIdx) const;

private:
  const Lattice&                  d_lattice;
  const ParticleVariable<int>&    d_pIsBroken;
  const ParticleVariable<Vector>& d_pCrackSurfaceNormal;
  const ParticleVariable<double>& d_pMicrocrackSize;
  const ParticleVariable<double>& d_pMicrocracPosition;
};
} // End namespace Uintah


#endif //__Packages/Uintah_MPM_BrokenCellShapeFunction__


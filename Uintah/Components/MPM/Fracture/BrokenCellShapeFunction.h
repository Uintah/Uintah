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

        BrokenCellShapeFunction( const Lattice& lattice,
                                 const ParticleVariable<double>& pVolume,
                                 const ParticleVariable<int>& pIsBroken,
				 const ParticleVariable<Vector>& pCrackSurfaceNormal,
				 const ParticleVariable<double>& pMicrocrackSize );

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
  const ParticleVariable<double>& d_pVolume;
  const ParticleVariable<int>&    d_pIsBroken;
  const ParticleVariable<Vector>& d_pCrackSurfaceNormal;
  const ParticleVariable<double>& d_pMicrocrackSize;
};

} //namespace MPM
} //namespace Uintah

#endif //__Uintah_MPM_BrokenCellShapeFunction__

// $Log$
// Revision 1.6  2000/09/08 18:24:39  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.5  2000/09/07 21:11:10  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.4  2000/09/05 07:44:17  tan
// Applied BrokenCellShapeFunction to constitutive models where fracture
// is involved.
//
// Revision 1.3  2000/09/05 06:59:15  tan
// Applied BrokenCellShapeFunction to SerialMPM::interpolateToParticlesAndUpdate
// where fracture is involved.
//
// Revision 1.2  2000/09/05 06:34:42  tan
// Introduced BrokenCellShapeFunction for SerialMPM::interpolateParticlesToGrid
// where farcture is involved.
//
// Revision 1.1  2000/08/11 03:13:30  tan
// Created BrokenCellShapeFunction to handle Shape functions (including Derivatives)
// for a cell containing cracked particles.
//

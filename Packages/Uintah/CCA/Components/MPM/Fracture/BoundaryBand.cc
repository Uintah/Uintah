#include "BoundaryBand.h"

#include "ParticlesNeighbor.h"
#include "CellsNeighbor.h"
#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include <Core/Exceptions/InternalError.h>

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

#include <iostream>
#include <float.h>  // for DBL_MAX

namespace Uintah {
using namespace SCIRun;

BoundaryBand::BoundaryBand()
{}

void BoundaryBand::setup(particleIndex pIdx,
	             const ParticleVariable<Vector>& pCrackNormal,
		     const ParticleVariable<int>& pIsBroken,
                     const ParticleVariable<double>& pVolume,
		     const Lattice& lattice,
		     double range)
{
  d_pCrackNormal = &pCrackNormal;
  d_pX = &lattice.getpX();
  d_pVolume = &pVolume;
  d_pIsBroken = &pIsBroken;

  IntVector cellIdx;
  lattice.getPatch()->findCell( (*d_pX)[pIdx],cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,lattice);

  int particlesNumber = particles.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx = particles[i];
    if(pidx == pIdx) continue;
    if( !pIsBroken[pidx] ) continue;
    Vector dis = (*d_pX)[pidx]-(*d_pX)[pIdx];
    if( Dot( pCrackNormal[pidx], dis ) < 0 ) continue;
    if( dis.length() < range ) {
      d_pIndexs.push_back(pidx);
    }
  }
  
  d_idx = pIdx;
}

void BoundaryBand::setup(const Point& p,
	             const ParticleVariable<Vector>& pCrackNormal,
		     const ParticleVariable<int>& pIsBroken,
                     const ParticleVariable<double>& pVolume,
		     const Lattice& lattice,
		     double range)
{
  d_pCrackNormal = &pCrackNormal;
  d_pX = &lattice.getpX();
  d_pVolume = &pVolume;
  d_pIsBroken = &pIsBroken;

  IntVector cellIdx;
  lattice.getPatch()->findCell(p,cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,lattice);

  int particlesNumber = particles.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx = particles[i];
    if( !pIsBroken[pidx] ) continue;
    Vector dis = (*d_pX)[pidx]-p;
    if( Dot( pCrackNormal[pidx], dis ) < 0 ) continue;
    if( dis.length() < range ) {
      d_pIndexs.push_back(pidx);
    }
  }

  d_idx = -1;
}

int BoundaryBand::inside(particleIndex pIdx) const
{
  if( pIdx == d_idx && (*d_pIsBroken)[pIdx] ) return 1;
  
  const Point& p = (*d_pX)[pIdx];
  
  int particlesNumber = d_pIndexs.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx = d_pIndexs[i];
    
    Point pCrack = (*d_pX)[pidx];
    if( Dot( (*d_pCrackNormal)[pidx], 
              p - pCrack ) < 0) return 1;
  }
  return 0;
}

int BoundaryBand::inside(const Point& p) const
{
  int particlesNumber = d_pIndexs.size();
  for(int i=0; i<particlesNumber; i++) {
    int pidx = d_pIndexs[i];
    Point pCrack = (*d_pX)[pidx];
    if( Dot( (*d_pCrackNormal)[pidx], 
              p - pCrack ) < 0) return 1;
  }
  return 0;
}

int BoundaryBand::numCracks() const
{
  return d_pIndexs.size();
}

} // End namespace Uintah

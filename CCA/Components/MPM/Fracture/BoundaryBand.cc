#include "BoundaryBand.h"

#include "ParticlesNeighbor.h"
#include "CellsNeighbor.h"
#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include "NormalFracture.h"

#include <Core/Exceptions/InternalError.h>

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

#include <iostream>
#include <float.h>  // for DBL_MAX

namespace Uintah {
using namespace SCIRun;

double BoundaryBand::d_correlate_cosine = 0.866;

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
    if( !pIsBroken[pidx] ) continue;
    Vector dis = (*d_pX)[pidx]-(*d_pX)[pIdx];
    if( Dot( pCrackNormal[pidx], dis ) < 0 ) continue;
    if( dis.length() < range ) {
      d_pIndexs.push_back(pidx);
    }
  }

  //walls
  std::vector<double> lengths;
  int numCrack = d_pIndexs.size();
  for(int i=0;i<numCrack;++i) {
    int numWall = d_wallDir.size();
    int pidx = d_pIndexs[i];
    const Vector& N = (*d_pCrackNormal)[pidx];

    bool newWall = true;
    for(int j=0;j<numWall;++j) {
      if( Dot(d_wallDir[j],N) / lengths[j] > d_correlate_cosine) {
        d_wallDir[j] += N;
	lengths[j] = d_wallDir[j].length();
        newWall = false;
        break;
      }
    }
    if(newWall) {
      d_wallDir.push_back(N);
      lengths.push_back(1.);
    }
  }

  int numWall = d_wallDir.size();
  for(int j=0;j<numWall;++j) {
    d_wallDir[j] /= lengths[j];
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

  //walls
  std::vector<double> lengths;
  int numCrack = d_pIndexs.size();
  for(int i=0;i<numCrack;++i) {
    int numWall = d_wallDir.size();
    int pidx = d_pIndexs[i];
    const Vector& N = (*d_pCrackNormal)[pidx];

    bool newWall = true;
    for(int j=0;j<numWall;++j) {
      if( Dot(d_wallDir[j],N) / lengths[j] > d_correlate_cosine) {
        d_wallDir[j] += N;
	lengths[j] = d_wallDir[j].length();
        newWall = false;
        break;
      }
    }
    if(newWall) {
      d_wallDir.push_back(N);
      lengths.push_back(1.);
    }
  }

  int numWall = d_wallDir.size();
  for(int j=0;j<numWall;++j) {
    d_wallDir[j] /= lengths[j];
  }

  d_idx = -1;
}

int BoundaryBand::inside(particleIndex pIdx) const
{
  int numCrack = d_pIndexs.size();

  for(int i=0; i<numCrack; i++) {
    int pidx = d_pIndexs[i];
    if(pidx==pIdx) return 1;
  }
  
  const Point& p = (*d_pX)[pIdx];
  
  int numWall = d_wallDir.size();
  std::vector<bool> inside(numWall, false);

  for(int i=0; i<numCrack; i++) {
    int pidx = d_pIndexs[i];
    double d = NormalFracture::connectionRadius((*d_pVolume)[pidx]);
    const Point& pCrack = (*d_pX)[pidx];
    const Vector& N = (*d_pCrackNormal)[pidx];
    Vector dis = p - pCrack;
    double vdis = Dot( (*d_pCrackNormal)[pidx], dis );
    if( (dis-(*d_pCrackNormal)[pidx]*vdis).length() < d) {
      if( vdis < 0 ) {
        for(int j=0;j<numWall;j++) {
	  if( Dot(N,d_wallDir[j]) > 0.866) {
	    inside[j] = true;
	    break;
	  }
	}
      }
    }
  }

  for(int j=0;j<numWall;j++) {
    if( !inside[j] ) return 0;
  }
  return 1;
}

int BoundaryBand::inside(const Point& p) const
{
  int numWall = d_wallDir.size();
  std::vector<bool> inside(numWall, false);

  int numCrack = d_pIndexs.size();
  for(int i=0; i<numCrack; i++) {
    int pidx = d_pIndexs[i];
    double d = NormalFracture::connectionRadius((*d_pVolume)[pidx]);
    const Point& pCrack = (*d_pX)[pidx];
    const Vector& N = (*d_pCrackNormal)[pidx];
    Vector dis = p - pCrack;
    double vdis = Dot( N, dis );
    if( (dis-N*vdis).length() < d) {
      if( vdis < 0 ) {
        for(int j=0;j<numWall;j++) {
	  if( Dot(N,d_wallDir[j]) > 0.866) {
	    inside[j] = true;
	    break;
	  }
	}
      }
    }
  }
  
  for(int j=0;j<numWall;j++) {
    if( !inside[j] ) return 0;
  }
  return 1;
}

int BoundaryBand::numCracks() const
{
  return d_pIndexs.size();
}

} // End namespace Uintah

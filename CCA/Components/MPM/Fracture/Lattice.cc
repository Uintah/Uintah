#include "Lattice.h"

#include "Cell.h"
#include "ParticlesNeighbor.h"

#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {

Lattice::
Lattice(const ParticleVariable<Point>& pX)
: Array3<Cell>( pX.getParticleSubset()->getPatch()->getCellLowIndex()
                  - IntVector(2,2,2),
                pX.getParticleSubset()->getPatch()->getCellHighIndex()
		  + IntVector(2,2,2) ),
    d_patch( pX.getParticleSubset()->getPatch() ),
    d_pX(&pX)
{
  ParticleSubset* pset = pX.getParticleSubset();

  for(ParticleSubset::iterator part_iter = pset->begin();
      part_iter != pset->end(); part_iter++)
  {
    (*this)[ d_patch->getLevel()->getCellIndex(pX[*part_iter]) ].
      insert(*part_iter);
  }
}

/*
Lattice::
Lattice(const Patch* patch)
: Array3<Cell>( patch->getCellLowIndex() - IntVector(2,2,2),
                patch->getCellHighIndex() + IntVector(2,2,2) ),
    d_patch( patch ),
    d_pX(NULL)
{
}
*/

void Lattice::insert(const CrackFace& crackFace)
{
  (*this)[ d_patch->getLevel()->getCellIndex(crackFace.getTip()) ].
    insert(crackFace);
}

bool Lattice::containCell(const IntVector& cellIndex) const
{
  if( cellIndex.x() >= getLowIndex().x() &&
      cellIndex.y() >= getLowIndex().y() &&
      cellIndex.z() >= getLowIndex().z() &&
      cellIndex.x() < getHighIndex().x() &&
      cellIndex.y() < getHighIndex().y() &&
      cellIndex.z() < getHighIndex().z() ) return true;
  else return false;
}

const Patch* Lattice::getPatch() const
{
  return d_patch;
}

const ParticleVariable<Point>& Lattice::getpX() const
{
  ASSERT(d_pX != NULL);
  return *d_pX;
}

void Lattice::getParticlesNeighbor(const Point& p, ParticlesNeighbor& particles)
{
  particles.clear();
  IntVector cellIdx;
  d_patch->findCell(p,cellIdx);
  particles.buildIn(cellIdx,*this);
}

bool Lattice::checkPossible(const Vector& N,
                   particleIndex thisIdx,
                   const ParticleVariable<Point>& pX,
                   const ParticleVariable<double>& pVolume,
                   const ParticleVariable<Vector>& pCrackNormal,
                   const ParticleVariable<int>& pIsBroken) const
{
  return true;
  IntVector cellIdx;
  d_patch->findCell(pX[thisIdx],cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,*this);

  double r = pow( pVolume[thisIdx],1./3.) / 2;
  double hRange = r * 2.2;
  
  for(int i=0; i<particles.size(); i++) {
    particleIndex idx = particles[i];
    if(idx == thisIdx ) continue;
    if( pIsBroken[idx] == 0 ) continue;
    
    Vector dis = pX[idx] - pX[thisIdx];
    double Vdis = Dot(dis, N );
    double Hdis = sqrt( dis.length2() - Vdis * Vdis );
    
    if(Hdis > hRange ) continue;
    
    if( fabs(Vdis) < r ) {
      if( Dot( N, pCrackNormal[idx] ) < 0 ) 
        return false;
    }

    if( Vdis > r*1.5 && Vdis < r*3 ) {
      if( Dot( N, pCrackNormal[idx] ) > 0 ) 
        return false;
    }
  }
  
  return true;
}

bool Lattice::checkPossible(
                   particleIndex thisIdx,
		   double r,
                   const ParticleVariable<Point>& pX,
                   const ParticleVariable<Vector>& pNewCrackNormal,
                   const ParticleVariable<int>& pIsBroken,
                   const ParticleVariable<int>& pNewIsBroken ) const
{
  return true;
  IntVector cellIdx;
  d_patch->findCell(pX[thisIdx],cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,*this);

  const Vector& N = pNewCrackNormal[thisIdx];
  
  for(int i=0; i<particles.size(); i++) {
    particleIndex idx = particles[i];
    if(idx == thisIdx ) continue;
    
    if( pIsBroken[idx] != 0 ) {
      Vector dis = pX[idx] - pX[thisIdx];
      double Vdis = Dot(dis, N );
      double Hdis = sqrt( dis.length2() - Vdis * Vdis );
      if(Hdis <= r*2.2) {
        if( Vdis > 1.5 && Vdis < r*2.2 ) {
          if( Dot( N, pNewCrackNormal[idx] ) < 0 ) 
          return true;
        }
      }
    }
    
    if( pNewIsBroken[idx] != 0 ) {
      Vector dis = pX[idx] - pX[thisIdx];
      double Vdis = Dot(dis, N );
      double Hdis = sqrt( dis.length2() - Vdis * Vdis );
      if(Hdis <= r*2.2) {
        if( Vdis > 1.5 && Vdis < r*2.2 ) {
          if( Dot( N, pNewCrackNormal[idx] ) < 0 ) 
          return true;
        }
      }
    }
  }
  return false;
}

void fit(ParticleSubset* pset_patchOnly,
	 const ParticleVariable<Point>& pX_patchOnly,
         ParticleSubset* pset_patchAndGhost,
	 const ParticleVariable<Point>& pX_patchAndGhost,
	 vector<int>& particleIndexExchange)
{
  Vector d = pset_patchOnly->getPatch()->dCell()/10;
  
  for(ParticleSubset::iterator iter_patchOnly = pset_patchOnly->begin();
       iter_patchOnly != pset_patchOnly->end(); iter_patchOnly++)
  {
    const Point& v = pX_patchOnly[*iter_patchOnly];
    for(ParticleSubset::iterator iter_patchAndGhost = pset_patchAndGhost->begin();
         iter_patchAndGhost != pset_patchAndGhost->end(); iter_patchAndGhost++)
    {
      const Point& p = pX_patchAndGhost[*iter_patchAndGhost];
      if( fabs(v.x()-p.x()) < d.x() && 
          fabs(v.y()-p.y()) < d.y() && 
          fabs(v.z()-p.z()) < d.z() )
      {
        particleIndexExchange[*iter_patchOnly] = *iter_patchAndGhost;
	break;
      }
    }
  }
}

} // End namespace Uintah
  


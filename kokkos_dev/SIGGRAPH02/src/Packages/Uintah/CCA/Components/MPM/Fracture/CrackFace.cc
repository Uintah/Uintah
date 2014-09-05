#include "CrackFace.h"

#include "Lattice.h"
#include "ParticlesNeighbor.h"
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>

#include <iostream>

namespace Uintah {

using namespace std;

void CrackFace::setup(const Vector& n,const Point& p,double halfGap)
{
  _normal = n;
  _tip = p;
  _halfGap = halfGap;
}

const Point& CrackFace::getTip() const
{
  return _tip;
}

void CrackFace::setTip(const Point& p)
{
 _tip = p;
}

const Vector& CrackFace::getNormal() const
{
  return _normal;
}

void CrackFace::setNormal(const Vector& n)
{
  _normal = n;
}

double CrackFace::getHalfGap() const
{
  return _halfGap;
}

double CrackFace::distance(const Point& p)
{
  return fabs( Dot( p-_tip, _normal ) );
}

bool CrackFace::isTip(const ParticleVariable<Vector>& pCrackNormal,
                      const ParticleVariable<int>& pIsBroken,
                      const Lattice& lattice) const
{
  IntVector cellIdx;
  lattice.getPatch()->findCell(_tip,cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,lattice);
  int particlesNumber = particles.size();
  
  std::vector<Vector> ds(particlesNumber);
  for(int j=0; j<particlesNumber; j++) {
    int idx = particles[j];
    const Point& p = lattice.getpX()[idx];
    ds[j] = p - _tip;
  };
  
  double hRange = _halfGap * 2.2;
  double vRange = _halfGap * 1.1;
  
  for(int j=0; j<particlesNumber; j++) {
    if( pIsBroken[particles[j]] != 0 ) continue;    

    double vdis = Dot( ds[j] , _normal);
    if( fabs(vdis) > vRange ) continue;
    if( (ds[j] - _normal * vdis ).length() > hRange ) continue;

    return true;
  }
  return false;
}

bool CrackFace::atTip(const Point& p) const
{
  double hRange = _halfGap * 2.2;
  double vRange = _halfGap * 1.1;
  
  Vector d = p - _tip;
  double vdis = Dot( d , _normal);
    
  if( fabs(vdis) > vRange ) return false;
    
  if( (d - _normal * vdis).length() > hRange ) return false;
  return true;
}

bool CrackFace::closeToBoundary(const Point& p,
                      const Lattice& lattice) const
{
  double crackHalfThick = _halfGap * 1.5;
  
  IntVector cellIdx;
  lattice.getPatch()->findCell(_tip,cellIdx);

  ParticlesNeighbor particles;
  particles.buildIn(cellIdx,lattice);
  int particlesNumber = particles.size();

  Vector N = Cross(_normal,p-_tip);
  N.normalize();

  int num = 0;
  for(int i=0; i<particlesNumber; i++) {
  if( fabs( Dot(N,lattice.getpX()[particles[i]] - _tip ) )
      < crackHalfThick ) num++;
  }

  if(num <= 34) return true;

  return false;
}


} //namespace

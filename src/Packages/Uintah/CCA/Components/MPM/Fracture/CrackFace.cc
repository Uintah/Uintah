#include "CrackFace.h"

#include "Lattice.h"
#include "ParticlesNeighbor.h"
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

const Matrix3& CrackFace::getStress() const
{
  return _stress;
}

void CrackFace::setStress(const Matrix3& s)
{
  _stress = s;
}

double CrackFace::getHalfGap() const
{
  return _halfGap;
}

const Vector& CrackFace::getMaxDirection() const
{
  return _maxDirection;
}

void CrackFace::computeMaxDirection()
{
  double sig[3];
  _stress.getEigenValues(sig[0], sig[1], sig[2]);
  double maxStress = sig[0];

  vector<Vector> eigenVectors = _stress.getEigenVectors(maxStress,
	   fabs(maxStress));

  for(int i=0;i<eigenVectors.size();++i) eigenVectors[i].normalize();

  if(eigenVectors.size() == 1) _maxDirection = eigenVectors[0];

  else if(eigenVectors.size() == 2) {
    cout<<"eigenVectors.size = 2"<<endl;
    double theta = drand48() * M_PI * 2;
    _maxDirection = eigenVectors[0] * cos(theta) + 
                    eigenVectors[1] * sin(theta);
  }

  else if(eigenVectors.size() == 3) {
    cout<<"eigenVectors.size = 3"<<endl;
    double theta = drand48() * M_PI * 2;
    double beta = drand48() * M_PI;
    double cos_beta = cos(beta);
    double sin_beta = sin(beta);
    Vector xy = eigenVectors[2] * sin_beta;
    _maxDirection = 
	     eigenVectors[0] * (sin_beta * cos(theta)) +
	     eigenVectors[1] * (sin_beta * sin(theta)) +
	     eigenVectors[2] * cos_beta;
  }
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

/*
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
  double crackHalfThick = _halfGap * 1.5;
  
  for(int j=0; j<particlesNumber; j++) {
    if( pIsBroken[particles[j]] != 0 ) continue;    

    double vdis = Dot( ds[j] , _normal);
    if( fabs(vdis) > vRange ) continue;
    if( (ds[j] - _normal * vdis ).length() > hRange ) continue;

    Vector N = Cross(_normal,ds[j]);
    N.normalize();
    int num = 0;
    for(int i=0; i<particlesNumber; i++) {
      if( fabs( Dot(N,ds[i]) ) < crackHalfThick ) num++;
    }
    //cout<<"tip neighbor particles number: "<<num<<endl;
    if(num > 34) return true;
    else return false;
  }
  return false;
}
*/

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

#include "SurfaceCouple.h"

#include "ParticlesNeighbor.h"
#include "CellsNeighbor.h"
#include "Cell.h"
#include "LeastSquare.h"
#include "NormalFracture.h"

#include <Core/Exceptions/InternalError.h>

#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

#include <iostream>
#include <float.h>  // for DBL_MAX

namespace Uintah {
using namespace SCIRun;

void SurfaceCouple::setup(particleIndex pIdxA,
                          particleIndex pIdxB,
			  const Vector& normal)
{
  d_pIdxA = pIdxA;
  d_pIdxB = pIdxB;
  d_normal = normal;
}

Point SurfaceCouple::crackTip(const ParticleVariable<Point>& pX) const
{
  return Point( (pX[d_pIdxA].x() + pX[d_pIdxB].x())/2,
                (pX[d_pIdxA].y() + pX[d_pIdxB].y())/2,
		(pX[d_pIdxA].z() + pX[d_pIdxB].z())/2 );
}

void SurfaceCouple::tipMatrix(const ParticleVariable<Matrix3>& pMatrix,
  Matrix3& matrix) const
{
  for(int i=1;i<=3;++i)
  for(int j=1;j<=3;++j) {
    matrix(i,j) = (pMatrix[d_pIdxA](i,j) + pMatrix[d_pIdxB](i,j))/2.;
  }
}

void SurfaceCouple::tipVector(const ParticleVariable<Vector>& pVector,
  Vector& vec) const
{
  for(int i=0;i<3;++i) {
    vec(i) = (pVector[d_pIdxA](i) + pVector[d_pIdxB](i))/2.;
  }
}

bool SurfaceCouple::extensible(
       particleIndex pIdx,
       const ParticleVariable<Point>& pX,
       const ParticleVariable<Vector>& pExtensionDirection,
       const ParticleVariable<Vector>& pCrackNormal,
       double volume,
       double& distanceToCrack) const
{
  Point tip(crackTip(pX));
  
  Vector dis = pX[pIdx] - tip;
  
  if( pExtensionDirection[d_pIdxA].length2() > 0.5 ) {
    if( Dot(dis,pExtensionDirection[d_pIdxA]) < 0 ) return false;
  }
  if( pExtensionDirection[d_pIdxB].length2() > 0.5 ) {
    if( Dot(dis,pExtensionDirection[d_pIdxB]) < 0 ) return false;
  }
  
  //if( Dot(dis,pExtensionDirection[d_pIdxA]) < 0 ) return false;
  //if( Dot(dis,pExtensionDirection[d_pIdxB]) < 0 ) return false;

  double r = pow(volume,0.333333)*0.866;
  double distance = dis.length();
  if(distance < distanceToCrack ) {
    double vDis = fabs( Dot(dis, d_normal) );
    if( vDis < r*1.1 ) {
      distanceToCrack = distance;
      return true;
    }
  }
  return false;
}

particleIndex SurfaceCouple::getIdxA() const
{
  return d_pIdxA;
}

particleIndex SurfaceCouple::getIdxB() const
{
  return d_pIdxB;
}

//TipNormal couples
bool SurfaceCouple::
computeCrackClosureIntegralAndCrackNormalFromForce(
	const Vector& nxx,
	const Lattice& lattice,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<Vector>& pDisplacement,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double toughness,
	double& GI,double& GII,double& GIII,Vector& N ) const
{
  static double Gmax = 0;

  double R = pow( (pVolume[d_pIdxA]+pVolume[d_pIdxB])/2, 0.333333 ) /2.;

  Vector ny = d_normal;
  Vector nz = Cross(nxx,ny);
  nz.normalize();
  Vector nx = Cross(ny,nz);
  
  Point pTip( crackTip( lattice.getpX() ) );
  pTip -= nx * (R/2);
  Vector dispTip( (pDisplacement[getIdxA()]+pDisplacement[getIdxB()]) /2 );

  Vector Dxy(0.,0.,0.);
  Vector DxY(0.,0.,0.);
  Vector DXy(0.,0.,0.);
  Vector DXY(0.,0.,0.);
  double volumexy = 0;
  double volumexY = 0;
  double volumeXy = 0;
  double volumeXY = 0;

  ParticlesNeighbor particles;
  lattice.getParticlesNeighbor(pTip, particles);

  int num = particles.size();
  for(int k=0; k<num; k++) {
    int index = particles[k];
    
    Vector d = lattice.getpX()[index] - pTip;
    double dx = Dot(d,nx);
    double dy = Dot(d,ny);

    if( d.length() < R * 3 ) {
      if(dy>0) {
        if( dx>0 ) {
	  DXY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXY += pVolume[index];
	}
        else {
	  DxY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexY += pVolume[index];
	}
      }
      else {
        if( dx>0 ) {
	  DXy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXy += pVolume[index];
	}
        else {
	  Dxy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexy += pVolume[index];
	}
      }
    }
  }

  if( volumeXY > 0 && volumexY > 0 && volumeXy > 0 && volumexy > 0 )
  {
    DXY /= volumeXY;
    DxY /= volumexY;
    DXy /= volumeXy;
    Dxy /= volumexy;
    
    double sigy1A = Dot( nx, pStress[d_pIdxA] * ny );
    double sigy2A = Dot( ny, pStress[d_pIdxA] * ny );
    double sigy3A = Dot( nz, pStress[d_pIdxA] * ny );
    double sigy1B = Dot( nx, pStress[d_pIdxB] * ny );
    double sigy2B = Dot( ny, pStress[d_pIdxB] * ny );
    double sigy3B = Dot( nz, pStress[d_pIdxB] * ny );
    
    Vector dispY = DxY-DXY;
    Vector dispy = Dxy-DXy;
    GI   = sigy2B * Dot(dispY,ny) - sigy2A * Dot(dispy,ny);
    GII  = sigy1B * Dot(dispY,nx) + sigy1A * Dot(dispy,nx);
    GIII = sigy3B * Dot(dispY,nz) + sigy3A * Dot(dispy,nz);
    
    double G = GI + GII + GIII;
    if(G>Gmax) {
      Gmax=G;
//      cout<<"Max energy release rate: "<<Gmax<<endl;
    }
    
    if( GI<0 || G<toughness ) return false;

    if(G < toughness) return false;
    
    Vector NA,NB;
    getMaxEigenvalue(pStress[d_pIdxA], NA);
    if( Dot(pTip-lattice.getpX()[d_pIdxA], NA) < 0 ) NA = -NA;
    getMaxEigenvalue(pStress[d_pIdxB], NB);
    if( Dot(pTip-lattice.getpX()[d_pIdxB], NB) < 0 ) NB = -NB;
    N = NA - NB;
    N.normalize();
    
    if(Dot(N, ny) < 0.5) return false;

    return true;
  }
  
  else return false;
}


#if 0
bool SurfaceCouple::
computeCrackClosureIntegralAndCrackNormalFromEnergyReleaseRate(
	const Vector& nxx,
	const Lattice& lattice,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<Vector>& pDisplacement,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double toughness,
	double& GI,double& GII,double& GIII,Vector& N ) const
{
  static double Gmax = 0;

  double R = pow( (pVolume[d_pIdxA]+pVolume[d_pIdxB])/2, 0.333333 ) /2.;

  Vector ny = d_normal;
  Vector nz = Cross(nxx,ny);
  nz.normalize();
  Vector nx = Cross(ny,nz);
  
  Point pTip( crackTip( lattice.getpX() ) );
  pTip -= nx * (R/2);
  Vector dispTip( (pDisplacement[getIdxA()]+pDisplacement[getIdxB()]) /2 );

  ParticlesNeighbor particles;
  lattice.getParticlesNeighbor(pTip, particles);

  int num = particles.size();

  int xBroken = -1;
  for(int k=0; k<num; k++) {
    int index = particles[k];
    Vector d = lattice.getpX()[index] - pTip;
    if( d.length() < R * 3 ) {
      double dx = Dot(d,nx);
      if(dx<0 && pIsBroken[index]) {
        xBroken = 1;
        break;
      }
    }
  }

  int XBroken = -1;
  for(int k=0; k<num; k++) {
    int index = particles[k];
    Vector d = lattice.getpX()[index] - pTip;
    if( d.length() < R * 3 ) {
      double dx = Dot(d,nx);
      if(dx>0 && pIsBroken[index]) {
        XBroken = 1;
        break;
      }
    }
  }
  
  if( XBroken * xBroken != -1 ) return false;

  Vector Dxy(0.,0.,0.);
  Vector DxY(0.,0.,0.);
  Vector DXy(0.,0.,0.);
  Vector DXY(0.,0.,0.);
  double volumexy = 0;
  double volumexY = 0;
  double volumeXy = 0;
  double volumeXY = 0;

  for(int k=0; k<num; k++) {
    int index = particles[k];
    
    Vector d = lattice.getpX()[index] - pTip;
    double dx = Dot(d,nx);
    double dy = Dot(d,ny);

    if( d.length() < R * 3 ) {
      if(dy>0) {
        if( dx>0 ) {
	  if(XBroken) DXY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXY += pVolume[index];
	}
        else {
	  if(xBroken) DxY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexY += pVolume[index];
	}
      }
      else {
        if( dx>0 ) {
	  if(XBroken) DXy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXy += pVolume[index];
	}
        else {
	  if(xBroken) Dxy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexy += pVolume[index];
	}
      }
    }
  }

  if( volumeXY > 0 && volumexY > 0 && volumeXy > 0 && volumexy > 0 )
  {
    DXY /= volumeXY;
    DxY /= volumexY;
    DXy /= volumeXy;
    Dxy /= volumexy;
    
    double sigy1A = Dot( nx, pStress[d_pIdxA] * ny );
    double sigy2A = Dot( ny, pStress[d_pIdxA] * ny );
    double sigy3A = Dot( nz, pStress[d_pIdxA] * ny );
    double sigy1B = Dot( nx, pStress[d_pIdxB] * ny );
    double sigy2B = Dot( ny, pStress[d_pIdxB] * ny );
    double sigy3B = Dot( nz, pStress[d_pIdxB] * ny );
    
    Vector dispY = DxY-DXY;
    Vector dispy = Dxy-DXy;
    GI   = sigy2B * Dot(dispY,ny) - sigy2A * Dot(dispy,ny);
    GII  = sigy1B * Dot(dispY,nx) + sigy1A * Dot(dispy,nx);
    GIII = sigy3B * Dot(dispY,nz) + sigy3A * Dot(dispy,nz);
    
    //cout<<"GII: "<<sigy1B * Dot(dispY,nx)<<"  "<<- sigy1A * Dot(dispy,nx)<<endl;

    double G = GI + GII + GIII;
    G= fabs(G);
    if(G>Gmax) {
      Gmax=G;
//      cout<<"Max energy release rate: "<<Gmax<<endl;
    }
    
    if( GI<0 || G<toughness ) return false;
    
    double theta = fabs( 2*sqrt( fabs(GII)/GI ) );
    double tau = (sigy1A+sigy1B)/2;
    if(tau>0)  theta = -theta;
    
    N = nx * (-sin(theta)) + ny * cos(theta);
    
    if(Dot(N, ny) < 0.5) return false;

    return true;
  }
  
  return false;
}
#endif


bool SurfaceCouple::
computeCrackClosureIntegralAndCrackNormalFromEnergyReleaseRate(
	const Vector& nxx,
	const Lattice& lattice,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<Vector>& pDisplacement,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double toughness,
	double& GI,double& GII,double& GIII,Vector& N ) const
{
  static double Gmax = 0;

  double R = pow( (pVolume[d_pIdxA]+pVolume[d_pIdxB])/2, 0.333333 ) /2.;

  Vector ny = d_normal;
  Vector nz = Cross(nxx,ny);
  nz.normalize();
  Vector nx = Cross(ny,nz);
  
  Point pTip( crackTip( lattice.getpX() ) );
  pTip -= nx * (R/2);
  Vector dispTip( (pDisplacement[getIdxA()]+pDisplacement[getIdxB()]) /2 );

  ParticlesNeighbor particles;
  lattice.getParticlesNeighbor(pTip, particles);

  int num = particles.size();

  int xBroken = 0;
  for(int k=0; k<num; k++) {
    int index = particles[k];
    Vector d = lattice.getpX()[index] - pTip;
    if( d.length() < R * 3 ) {
      double dx = Dot(d,nx);
      if(dx<0 && pIsBroken[index]) {
        xBroken = 1;
        break;
      }
    }
  }

  int XBroken = 0;
  for(int k=0; k<num; k++) {
    int index = particles[k];
    Vector d = lattice.getpX()[index] - pTip;
    if( d.length() < R * 3 ) {
      double dx = Dot(d,nx);
      if(dx>0 && pIsBroken[index]) {
        XBroken = 1;
        break;
      }
    }
  }
  
  if( (!XBroken) && (!xBroken) ) return false;
  if( XBroken && xBroken ) {
    XBroken = 0;
    xBroken = 1;
  }

  Vector Dxy(0.,0.,0.);
  Vector DxY(0.,0.,0.);
  Vector DXy(0.,0.,0.);
  Vector DXY(0.,0.,0.);
  double volumexy = 0;
  double volumexY = 0;
  double volumeXy = 0;
  double volumeXY = 0;

  for(int k=0; k<num; k++) {
    int index = particles[k];
    
    Vector d = lattice.getpX()[index] - pTip;
    double dx = Dot(d,nx);
    double dy = Dot(d,ny);

    if( d.length() < R * 3 ) {
      if(dy>0) {
        if( dx>0 ) {
	  if(XBroken==1) DXY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXY += pVolume[index];
	}
        else {
	  if(xBroken==1) DxY += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexY += pVolume[index];
	}
      }
      else {
        if( dx>0 ) {
	  if(XBroken==1) DXy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumeXy += pVolume[index];
	}
        else {
	  if(xBroken==1) Dxy += (pDisplacement[index]-dispTip) * pVolume[index];
	  volumexy += pVolume[index];
	}
      }
    }
  }

  if( volumeXY > 0 && volumexY > 0 && volumeXy > 0 && volumexy > 0 )
  {
    DXY /= volumeXY;
    DxY /= volumexY;
    DXy /= volumeXy;
    Dxy /= volumexy;
    
    double sigy1A = Dot( nx, pStress[d_pIdxA] * ny );
    double sigy2A = Dot( ny, pStress[d_pIdxA] * ny );
    double sigy3A = Dot( nz, pStress[d_pIdxA] * ny );
    double sigy1B = Dot( nx, pStress[d_pIdxB] * ny );
    double sigy2B = Dot( ny, pStress[d_pIdxB] * ny );
    double sigy3B = Dot( nz, pStress[d_pIdxB] * ny );
    
    Vector dispY = DxY-DXY;
    Vector dispy = Dxy-DXy;
    GI   = sigy2B * Dot(dispY,ny) - sigy2A * Dot(dispy,ny);
    GII  = sigy1B * Dot(dispY,nx) + sigy1A * Dot(dispy,nx);
    GIII = sigy3B * Dot(dispY,nz) + sigy3A * Dot(dispy,nz);
    
    //cout<<"GII: "<<sigy1B * Dot(dispY,nx)<<"  "<<- sigy1A * Dot(dispy,nx)<<endl;

    double G = GI + GII + GIII;
    G= fabs(G);
    if(G>Gmax) {
      Gmax=G;
//      cout<<"Max energy release rate: "<<Gmax<<endl;
    }
    
    if( GI<0 || G<toughness ) return false;
    
    double theta = fabs( 2*sqrt( fabs(GII)/GI ) );
    double tau = (sigy1A+sigy1B)/2;
    if(tau>0)  theta = -theta;
    
    N = nx * (-sin(theta)) + ny * cos(theta);
    
    if(Dot(N, ny) < 0.5) return false;

    return true;
  }
  
  return false;
}


} // End namespace Uintah

#include "ParticlesNeighbor.h"

#include "CellsNeighbor.h"
#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include <Core/Exceptions/InternalError.h>

#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

#include <iostream>

namespace Uintah {
using namespace SCIRun;

ParticlesNeighbor::ParticlesNeighbor()
: std::vector<particleIndex>()
{
}

void ParticlesNeighbor::buildIn(const IntVector& cellIndex,const Lattice& lattice)
{
  CellsNeighbor cellsNeighbor;
  cellsNeighbor.buildIncluding(cellIndex,lattice);
  
  for(CellsNeighbor::const_iterator iter_cell = cellsNeighbor.begin();
    iter_cell != cellsNeighbor.end();
    ++iter_cell )
  {
    std::vector<particleIndex>& parts = lattice[*iter_cell].particles;
    for( std::vector<particleIndex>::const_iterator iter_p = parts.begin();
         iter_p != parts.end();
         ++iter_p )
    {
      push_back(*iter_p);
    }
  }
}

/*
void  ParticlesNeighbor::interpolateVector(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<Vector>& pVector,
                          Vector& data,
                          Matrix3& gradient) const
{
  Vector v;
  for(int i=0;i<3;++i) {
    ls.clean();
    for(const_iterator pIter = begin(); pIter != end(); pIter++) {
      ls.input( (*d_pX)[*pIter]-(*d_pX)[pIdx], pVector[*pIter](i) );
    }
    ls.output( data(i),v );
    for(int j=0;j<3;++j) {
      gradient(i,j) = v(j);
    }
  }
}

void  ParticlesNeighbor::interpolatedouble(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<double>& pdouble,
                          double& data,
                          Vector& gradient) const
{
  ls.clean();
  for(const_iterator pIter = begin(); pIter != end(); pIter++) {
    ls.input( (*d_pX)[*pIter]-(*d_pX)[pIdx], pdouble[*pIter] );
  }
  ls.output( data,gradient );
}

void  ParticlesNeighbor::interpolateInternalForce(LeastSquare& ls,
                          const particleIndex& pIdx,
                          const ParticleVariable<Matrix3>& pStress,
                          Vector& pInternalForce) const
{
  Vector v;
  double data;
  for(int i=0;i<3;++i) {
    pInternalForce(i) = 0;
    for(int j=0;j<3;++j) {
      ls.clean();
      for(const_iterator pIter = begin(); pIter != end(); pIter++) {
        ls.input( (*d_pX)[*pIter]-(*d_pX)[pIdx], pStress[*pIter](i,j) );
      }
    }
    ls.output( data,v );
    pInternalForce(i) -= v(i);
  }
}
*/

bool ParticlesNeighbor::visible(particleIndex idx,
                                const Point& B,
				const ParticleVariable<Point>& pX,
				const ParticleVariable<int>& pIsBroken,
				const ParticleVariable<Vector>& pCrackSurfaceNormal1,
				const ParticleVariable<Vector>& pCrackSurfaceNormal2,
				const ParticleVariable<Vector>& pCrackSurfaceNormal3,
				const ParticleVariable<double>& pVolume) const
{
  const Point& A = pX[idx];
  if(pIsBroken[idx]>=1) 
    if( Dot( B - A, pCrackSurfaceNormal1[idx] ) > 0 ) return false;
  if(pIsBroken[idx]>=2)
    if( Dot( B - A, pCrackSurfaceNormal2[idx] ) > 0 ) return false;
  if(pIsBroken[idx]>=3)
    if( Dot( B - A, pCrackSurfaceNormal3[idx] ) > 0 ) return false;
  
  for(int i=0; i<(int)size(); i++) {
    int index = (*this)[i];
    if( index == idx ) continue;
      
    const Point& O = pX[index];
    double size2 = pow(pVolume[index] *0.75/M_PI,0.666666667);
    
    if(pIsBroken[index]>=1)
      if( !visible(A,B,O,pCrackSurfaceNormal1[index],size2) ) return false;
    if(pIsBroken[index]>=2)
      if( !visible(A,B,O,pCrackSurfaceNormal2[index],size2) ) return false;
    if(pIsBroken[index]>=3)
      if( !visible(A,B,O,pCrackSurfaceNormal3[index],size2) ) return false;
  }
  return true;
}

bool ParticlesNeighbor::visible(const Point& A,
                                const Point& B,
				const Point& O,const Vector& N,double size2) const
{
  double A_N = Dot(A,N);
	
  double a = A_N - Dot(O,N);
  double b = A_N - Dot(B,N);
	
  if(b != 0) {
    double lambda = a/b;
    if( lambda>=0 && lambda<=1 ) {
      Point p( A.x() * (1-lambda) + B.x() * lambda,
               A.y() * (1-lambda) + B.y() * lambda,
	       A.z() * (1-lambda) + B.z() * lambda );
      if( (p - O).length2() < size2 ) return false;
    }
  }
  return true;
}

bool ParticlesNeighbor::visible(particleIndex idxA,
                                particleIndex idxB,
				const ParticleVariable<Point>& pX,
				const ParticleVariable<int>& pIsBroken,
				const ParticleVariable<Vector>& pCrackSurfaceNormal1,
				const ParticleVariable<Vector>& pCrackSurfaceNormal2,
				const ParticleVariable<Vector>& pCrackSurfaceNormal3,
				const ParticleVariable<double>& pVolume) const
{
  const Point& A = pX[idxA];
  const Point& B = pX[idxB];
  Vector d = B - A;
  d.normalize();

  if(pIsBroken[idxA] >= 1)
    if( Dot( d, pCrackSurfaceNormal1[idxA] ) > 0.707 ) return false;
  if(pIsBroken[idxA] >= 2)
    if( Dot( d, pCrackSurfaceNormal2[idxA] ) > 0.707 ) return false;
  if(pIsBroken[idxA] >= 3)
    if( Dot( d, pCrackSurfaceNormal3[idxA] ) > 0.707 ) return false;

  if(pIsBroken[idxB] >= 1)
    if( Dot( d, pCrackSurfaceNormal1[idxB] ) < -0.707 ) return false;
  if(pIsBroken[idxB] >= 2)
    if( Dot( d, pCrackSurfaceNormal2[idxB] ) < -0.707 ) return false;
  if(pIsBroken[idxB] >= 3)
    if( Dot( d, pCrackSurfaceNormal3[idxB] ) < -0.707 ) return false;
  
  for(int i=0; i<(int)size(); i++) {
    int index = (*this)[i];
    if( index == idxA || index == idxB ) continue;
    if(pIsBroken[index] == 0) continue;

    const Point& O = pX[index];
    double size2 = pow(pVolume[index] *0.75/M_PI,0.666666667);
    
    if(pIsBroken[index]>=1)
      if( !visible(A,B,O,pCrackSurfaceNormal1[index],size2) ) return false;
    if(pIsBroken[index]>=2)
      if( !visible(A,B,O,pCrackSurfaceNormal2[index],size2) ) return false;
    if(pIsBroken[index]>=3)
      if( !visible(A,B,O,pCrackSurfaceNormal3[index],size2) ) return false;
  }
  return true;
}

bool ParticlesNeighbor::computeEnergyReleaseRate(
        particleIndex tipIndex,
        const Vector& nx,
	Vector& ny,
	double stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<double>& pVolume,
	double& G) const
{
  double volumea1=0;
  double volumea2=0;
  double opena1=0;
  double opena2=0;
  int numa1=0;
  int numa2=0;

  double volumeb1=0;
  double volumeb2=0;
  double openb1=0;
  double openb2=0;
  int numb1=0;
  int numb2=0;

  double volumec1=0;
  double volumec2=0;
  double openc1=0;
  double openc2=0;
  int numc1=0;
  int numc2=0;

  double volumed1=0;
  double volumed2=0;
  double opend1=0;
  double opend2=0;
  int numd1=0;
  int numd2=0;

  double psize = pow( pVolume[tipIndex],1./3. );
  Point pTipa = pX[tipIndex] + ny*(psize/2) + nx*(psize/2);
  Point pTipb = pX[tipIndex] + ny*(psize/2) - nx*(psize/2);
  Point pTipc = pX[tipIndex] - ny*(psize/2) + nx*(psize/2);
  Point pTipd = pX[tipIndex] - ny*(psize/2) - nx*(psize/2);
  
  int num = size();
  for(int i=0; i<num; i++) {
    int index = (*this)[i];
    
    Vector da = pX[index] - pTipa;
    Vector db = pX[index] - pTipb;
    Vector dc = pX[index] - pTipc;
    Vector dd = pX[index] - pTipd;

    double dxa = Dot(da,nx);
    double dxb = Dot(db,nx);
    double dxc = Dot(dc,nx);
    double dxd = Dot(dd,nx);

    double dya = Dot(da,ny);
    double dyb = Dot(db,ny);
    double dyc = Dot(dc,ny);
    double dyd = Dot(dd,ny);

    if( sqrt(dxa*dxa+dya*dya) < psize ) {
      if( dxa>0 ) {
        volumea1 += pVolume[index];
        opena1 += fabs(dya) * pVolume[index];
        numa1++;
      }
      if( dxa<0 ) {
        volumea2 += pVolume[index];
        opena2 += fabs(dya) * pVolume[index];
        numa2++;
      }
    }
    else if( sqrt(dxb*dxb+dyb*dyb) < psize ) {
      if( dxb>0 ) {
        volumeb1 += pVolume[index];
        openb1 += fabs(dyb) * pVolume[index];
        numb1++;
      }
      if( dxb<0 ) {
        volumeb2 += pVolume[index];
        openb2 += fabs(dyb) * pVolume[index];
        numb2++;
      }
    }
    else if( sqrt(dxc*dxc+dyc*dyc) < psize ) {
      if( dxc>0 ) {
        volumec1 += pVolume[index];
        openc1 += fabs(dyc) * pVolume[index];
        numc1++;
      }
      if( dxc<0 ) {
        volumec2 += pVolume[index];
        openc2 += fabs(dyc) * pVolume[index];
        numc2++;
      }
    }
    else if( sqrt(dxd*dxd+dyd*dyd) < psize ) {
      if( dxd>0 ) {
        volumed1 += pVolume[index];
        opend1 += fabs(dyd) * pVolume[index];
        numd1++;
      }
      if( dxd<0 ) {
        volumed2 += pVolume[index];
        opend2 += fabs(dyd) * pVolume[index];
        numd2++;
      }
    }
  }

  G = 0;
    
  if(numa1 > 0 && numa2 > 0) {
    opena1 /= volumea1;
    opena2 /= volumea2;
    double Ga=stress*fabs(opena1-opena2);
    if(Ga>G)G=Ga;
  }
  else if(numb1 > 0 && numb2 > 0) {
    openb1 /= volumeb1;
    openb2 /= volumeb2;
    double Gb=stress*fabs(openb1-openb2);
    if(Gb>G)G=Gb;
  }
  else if(numc1 > 0 && numc2 > 0) {
    openc1 /= volumec1;
    openc2 /= volumec2;
    double Gc=stress*fabs(openc1-openc2);
    if(Gc>G) {
      G=Gc;
      ny = -ny;
    }
  }
  else if(numd1 > 0 && numd2 > 0) {
    opend1 /= volumed1;
    opend2 /= volumed2;
    double Gd=stress*fabs(opend1-opend2);
    if(Gd>G) {
      G=Gd;
      ny = -ny;
    }
  }
  else return false;

  return true;
}

} // End namespace Uintah

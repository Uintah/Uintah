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
	const Vector& ny,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<Matrix3>& pStress,
	const ParticleVariable<double>& pVolume,
	double& G) const
{
  double volume1=0;
  double volume2=0;
  double open1=0;
  double open2=0;
  double stress1=0;
  double stress2=0;
  int num1=0;
  int num2=0;
  
  const Point& pTip = pX[tipIndex];
  
  int num = size();
  for(int i=0; i<num; i++) {
    int index = (*this)[i];
    if(tipIndex == index) continue;
    
    Vector d = pX[index] - pTip;
    double dx = Dot(d,nx);
    double dy = Dot(d,ny);
    
    if( dx>0 ) {
      volume1 += pVolume[index];
      open1 += fabs(dy) * pVolume[index];
      stress1 += Dot(ny, pStress[index] * ny) * pVolume[index];
      num1++;
    }
    if( dx<0 ) {
      volume2 += pVolume[index];
      open2 += fabs(dy) * pVolume[index];
      stress2 += Dot(ny, pStress[index] * ny) * pVolume[index];
      num2++;
    }
  }
  
  if(num1 == 0) return false;
  if(num2 == 0) return false;
  
  volume1 /= volume1;
  open1 /= volume1;
  stress1 /= volume1;

  volume2 /= volume2;
  open2 /= volume2;
  stress2 /= volume2;
  
  if(stress1<=0 && stress2<=0) return false;
  
  if(stress1>stress2) G=stress1*open2-stress2*open1;
  else G=stress2*open1-stress1*open2;
  
  return true;
}

} // End namespace Uintah

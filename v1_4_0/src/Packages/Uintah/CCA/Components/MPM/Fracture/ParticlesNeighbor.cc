#include "ParticlesNeighbor.h"

#include "CellsNeighbor.h"
#include "Lattice.h"
#include "Cell.h"
#include "LeastSquare.h"
#include <Core/Exceptions/InternalError.h>

#include <Packages/Uintah/Core/Math/Matrix3.h>

#include <Packages/Uintah/Core/Grid/Patch.h>

#include <iostream>
#include <float.h>  // for DBL_MAX

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
    const std::vector<particleIndex>& parts = lattice[*iter_cell].particles;
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

double ParticlesNeighbor::computeEnergyReleaseRate(
        particleIndex tipIndex,
        const Matrix3& stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<double>& pVolume,
	const ParticleVariable<int>& pIsBroken,
	double& sigmaN,
	Vector& Ny,
	Vector& Nx) const
{
  static double open_limit = DBL_MAX/2;

  Vector e[3];
  double sig[3];
  getEigenInfo(stress,e[0],sig[0],e[1],sig[1],e[2],sig[2]);
  
  double R = pow( pVolume[tipIndex],1./3. ) /2;

  double G = 0;

  for(int i=0;i<12;++i) {
    Vector nx,ny;
    double sigma;

    double openxy = DBL_MAX;
    double openxY = DBL_MAX;
    double openXy = DBL_MAX;
    double openXY = DBL_MAX;

    bool nearCracky = false;
    bool nearCrackY = false;
    
         if(i==0)  {nx= e[0];ny= e[1];sigma=sig[1];}
    else if(i==1)  {nx= e[0];ny= e[2];sigma=sig[2];}
    else if(i==2)  {nx= e[1];ny= e[0];sigma=sig[0];}
    else if(i==3)  {nx= e[1];ny= e[2];sigma=sig[2];}
    else if(i==4)  {nx= e[2];ny= e[1];sigma=sig[1];}
    else if(i==5)  {nx= e[2];ny= e[0];sigma=sig[0];}
    else if(i==6)  {nx= e[0];ny=-e[1];sigma=sig[1];}
    else if(i==7)  {nx= e[0];ny=-e[2];sigma=sig[2];}
    else if(i==8)  {nx= e[1];ny=-e[0];sigma=sig[0];}
    else if(i==9)  {nx= e[1];ny=-e[2];sigma=sig[2];}
    else if(i==10) {nx= e[2];ny=-e[1];sigma=sig[1];}
    else           {nx= e[2];ny=-e[0];sigma=sig[0];}

    Point pTip = pX[tipIndex] + ny*(R) + nx*(R);

    int num = size();
    for(int k=0; k<num; k++) {
      int index = (*this)[k];

      Vector d = pX[index] - pTip;

      double r = pow( pVolume[index],0.333333 ) /2;

      double dx = Dot(d,nx);
      double dy = Dot(d,ny);

      if( d.length() < R * 1.5 ) {
        if(dy>0) {
          if(pIsBroken[index]>0) nearCrackY = true;
          if( dx>0 ) {
            openXY = Min(dy-r,openXY);
          }
          else {
            openxY = Min(dy-r,openxY);
          }
	}
        else {
          if(pIsBroken[index]>0) nearCracky = true;
          if( dx>0 ) {
            openXy = Min(-dy-r,openXy);
          }
          else {
            openxy = Min(-dy-r,openxy);
          }
	}
      }
    }
    
    if( nearCracky && nearCrackY && 
        openXY < open_limit &&
	openxY < open_limit &&
	openXy < open_limit &&
	openxy < open_limit )
    {
      double openx = openxY + openxy;
      double openX = openXY + openXy;
      if(openx<0)openx=0;
      if(openX<0)openX=0;
      
      double g = sigma * fabs(openx-openX);
      if(g>G) {
        G = g;
	Ny = ny;
	Nx = nx;
	sigmaN = sigma;
      }
    }
  }
  
  return G;
}

double ParticlesNeighbor::computeCrackClosureIntegral(
        const Point& pTip,
	double R,
	const Vector& nx,
	Vector ny,
        const Matrix3& stress,
	const ParticleVariable<Point>& pX,
	const ParticleVariable<Vector>& pVelocity,
	const ParticleVariable<double>& pVolume,
	double delT ) const
{
  ny = -ny;
  
  Vector Vxy,VxY,VXy,VXY;
  double volumexy = 0;
  double volumexY = 0;
  double volumeXy = 0;
  double volumeXY = 0;

  int num = size();
  for(int k=0; k<num; k++) {
    int index = (*this)[k];

    Vector d = pX[index] - pTip;
    double dx = Dot(d,nx);
    double dy = Dot(d,ny);

    if( d.length() < R * 3 ) {
      if(dy>0) {
        if( dx>0 ) {
	  VXY += pVelocity[index] * pVolume[index];
	  volumeXY += pVolume[index];
	}
        else {
	  VxY += pVelocity[index] * pVolume[index];
	  volumexY += pVolume[index];
	}
      }
      else {
        if( dx>0 ) {
	  VXy += pVelocity[index] * pVolume[index];
	  volumeXy += pVolume[index];
	}
        else {
	  Vxy += pVelocity[index] * pVolume[index];
	  volumexy += pVolume[index];
	}
      }
    }
  }
  
  double G = 0;
  if( volumeXY > 0 && volumexY > 0 && volumeXy > 0 && volumexy > 0 )
  {
    VXY /= volumeXY;
    VxY /= volumexY;
    VXy /= volumeXy;
    Vxy /= volumexy;
    
    Vector Vdiff = (VxY - Vxy) - (VXY - VXy);

    Vector nz = Cross(nx,ny);
    double sigy1 = Dot( nx, stress * ny );
    double sigy2 = Dot( ny, stress * ny );
    double sigy3 = Dot( nz, stress * ny );
    
    double G1 = sigy1 * Dot(Vdiff,nx) * delT;
    double G2 = sigy2 * Dot(Vdiff,ny) * delT;
    double G3 = sigy3 * Dot(Vdiff,nz) * delT;
    
    //cout<<"G1: "<<G1<<" G2: "<<G2<<" G3: "<<G3<<endl;
    
    G = G1 + G2 + G3;
  }
  
  if(G>0) return G;
  else return 0;
}

} // End namespace Uintah

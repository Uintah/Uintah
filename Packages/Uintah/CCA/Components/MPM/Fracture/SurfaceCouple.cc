#include "SurfaceCouple.h"

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

void SurfaceCouple::setup(particleIndex pIdxA,
                          particleIndex pIdxB,
			  const Vector& normal)
{
  d_pIdxA = pIdxA;
  d_pIdxB = pIdxB;
  d_normal = normal;
}

bool SurfaceCouple::extensible(
       particleIndex pIdx,
       const ParticleVariable<Point>& pX,
       const ParticleVariable<Vector>& pExtensionDirection,
       double volume,
       double& distanceToCrack) const
{
  Point tip = Point( (pX[d_pIdxA].x() + pX[d_pIdxB].x())/2,
                     (pX[d_pIdxA].y() + pX[d_pIdxB].y())/2,
		     (pX[d_pIdxA].z() + pX[d_pIdxB].z())/2 );

  Vector dis = pX[pIdx] - tip;
  if( Dot(dis,pExtensionDirection[d_pIdxA]) < 0 ) return false;
  if( Dot(dis,pExtensionDirection[d_pIdxB]) < 0 ) return false;

  double r = pow(volume,0.333333)/2;
  if(dis.length() > r*3) return false;
  
  double vDis = fabs( Dot(dis, d_normal) );
  if( vDis < distanceToCrack ) {
    distanceToCrack = vDis;
  }
  return true;
}

} // End namespace Uintah

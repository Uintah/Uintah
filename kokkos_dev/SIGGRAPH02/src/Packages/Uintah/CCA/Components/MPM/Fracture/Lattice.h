#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Core/Geometry/Point.h>
#include "Cell.h"

namespace Uintah {

using namespace SCIRun;
class ParticlesNeighbor;

class Lattice : public Array3<Cell> {
public:
        Lattice(const ParticleVariable<Point>& pX);

  void  insert(const CrackFace& crackFace);
  
  bool              containCell(const IntVector& cellIndex) const;
  
  const Patch*                    getPatch() const;
  const ParticleVariable<Point>&  getpX() const;

  void  getParticlesNeighbor(const Point& p, 
                             ParticlesNeighbor& particles) const;

  bool  checkPossible(const Vector& N,
                   particleIndex thisIdx,
                   const ParticleVariable<Point>& pX,
                   const ParticleVariable<double>& pVolume,
                   const ParticleVariable<Vector>& pCrackNormal,
                   const ParticleVariable<int>& pIsBroken) const;

  bool  checkPossible(
                   particleIndex thisIdx,
		   double r,
                   const ParticleVariable<Point>& pX,
                   const ParticleVariable<Vector>& pNewCrackNormal,
                   const ParticleVariable<int>& pIsBroken,
                   const ParticleVariable<int>& pNewIsBroken ) const;

private:
  const Patch*                   d_patch;
  const ParticleVariable<Point>* d_pX;
};

} // End namespace Uintah

#endif //__LATTICE_H__


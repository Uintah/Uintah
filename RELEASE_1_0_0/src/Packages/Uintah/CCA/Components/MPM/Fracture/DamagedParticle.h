#ifndef _MPM_DamagedParticle
#define _MPM_DamagedParticle

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Core/Geometry/Point.h>

namespace Uintah {
using namespace SCIRun;

class Cell;
class Matrix3;

class DamagedParticle {
public:
  double           CriticalStressIntensityFactor() const;

                   DamagedParticle( const Matrix3& stress,
                                    double averageMicrocrackLength,
                                    double toughness,
                                    double youngModulus,
                                    double poissonRatio
                                  );

private:
  const Point*     d_position;
  const Matrix3&   d_stress;
  double           d_volume;
  double           d_mass;
  double           d_averageMicrocrackLength;
  double           d_toughness;
  double           d_youngModulus;
  double           d_poissonRatio;
};
} // End namespace Uintah


#endif //Packages/Uintah_MPM_DamagedParticle


#include "DamagedParticle.h"

#include "Packages/Uintah/Core/Math/Matrix3.h"

namespace Uintah {
DamagedParticle::DamagedParticle( const Matrix3& stress,
                                  double averageMicrocrackLength,
                                  double toughness,
                                  double youngModulus,
                                  double poissonRatio
                                )
: d_stress(stress),
  d_averageMicrocrackLength(averageMicrocrackLength),
  d_toughness(toughness),
  d_youngModulus(youngModulus),
  d_poissonRatio(poissonRatio)
{
}

double DamagedParticle::CriticalStressIntensityFactor() const
{
  return sqrt( d_toughness * d_youngModulus / 
               (1 - d_poissonRatio*d_poissonRatio) );
}

} // End namespace Uintah
  


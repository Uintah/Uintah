#include "DamagedParticle.h"

#include "Uintah/Components/MPM/Util/Matrix3.h"

namespace Uintah {
namespace MPM {

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

  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.1  2000/06/06 02:49:21  tan
// Created class DamagedParticle to handle particle spliting when crack
// pass through.
//

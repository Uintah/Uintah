#ifndef Uintah_MPM_DamagedParticle
#define Uintah_MPM_DamagedParticle

#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Array3.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

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

} //namespace MPM
} //namespace Uintah

#endif //Uintah_MPM_DamagedParticle

// $Log$
// Revision 1.1  2000/06/06 02:49:10  tan
// Created class DamagedParticle to handle particle spliting when crack
// pass through.
//

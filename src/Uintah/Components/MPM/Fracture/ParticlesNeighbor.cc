#include "ParticlesNeighbor.h"

#include <Uintah/Grid/Patch.h>

namespace Uintah {
namespace MPM {

void
ParticlesNeighbor::
interpolateVector( const ParticleVariable<Vector>& pVector,
                   Vector* vector, 
                   Matrix3* gradient) const
{
}

void
ParticlesNeighbor::
interpolatedouble(const ParticleVariable<double>& pdouble,
                  double* data,
                  Vector* gradient) const
{
}

} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.2  2000/06/05 22:30:11  tan
// Added interpolateVector and interpolatedouble for least-square approximation.
//
// Revision 1.1  2000/06/05 21:15:36  tan
// Added class ParticlesNeighbor to handle neighbor particles searching.
//

#include "Fracture.h"

namespace Uintah {
namespace MPM {

double
Fracture::
d_crackDensity;

double
Fracture::
d_averageMicrocrackLength;

double
Fracture::
d_materialToughness;

void
Fracture::
fractureParametersInitialize()
{
  d_crackDensity = 0.05;
  d_averageMicrocrackLength = 1e-6; //meter
  d_materialToughness = 1e6; //Pa sqrt(meter)
}

void
Fracture::
materialDefectsInitialize()
{
}
  
} //namespace MPM
} //namespace Uintah

#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
using namespace Uintah;

MPMFlags::MPMFlags()
{
  d_8or27 = 8;
  d_integrator_type = "explicit";

  d_artificial_viscosity = false;
  d_accStrainEnergy = false;
  d_useLoadCurves = false;
  d_createNewParticles = false;
                      
  d_doErosion = false;
  d_erosionAlgorithm = "none";

  d_adiabaticHeating = 0.0;
  d_artificialDampCoeff = 0.0;
  d_forceIncrementFactor = 1.0;
}

MPMFlags::~MPMFlags()
{
}


#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg("MPMFlags", false);

MPMFlags::MPMFlags()
{
  d_8or27 = 8;
  d_integrator_type = "explicit";

  d_artificial_viscosity = false;
  d_artificialViscCoeff1 = 0.2;
  d_artificialViscCoeff2 = 2.0;
  d_accStrainEnergy = false;
  d_useLoadCurves = false;
  d_createNewParticles = false;
  d_addNewMaterial = false;
  d_with_color = false;
  d_fracture = false;
  d_finestLevelOnly = false;
                      
  d_doErosion = false;
  d_erosionAlgorithm = "none";

  d_adiabaticHeating = 0.0;
  d_artificialDampCoeff = 0.0;
  d_forceIncrementFactor = 1.0;
}

MPMFlags::~MPMFlags()
{
}

void
MPMFlags::readMPMFlags(ProblemSpecP& ps)
{
  ps->get("nodes8or27", d_8or27);
  ps->get("withColor",  d_with_color);
  ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  ps->get("artificial_viscosity",     d_artificial_viscosity);
  ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  ps->get("accumulate_strain_energy", d_accStrainEnergy);
  ps->get("use_load_curves", d_useLoadCurves);
  bool adiabaticHeatingOn = true;
  ps->get("turn_on_adiabatic_heating", adiabaticHeatingOn);
  ps->get("finest_level_only", d_finestLevelOnly);
  if (!adiabaticHeatingOn) d_adiabaticHeating = 1.0;
  ps->get("ForceBC_force_increment_factor", d_forceIncrementFactor);
  ps->get("create_new_particles", d_createNewParticles);
  ps->get("manual_new_material", d_addNewMaterial);
  ProblemSpecP erosion_ps = ps->findBlock("erosion");
  if (erosion_ps) {
    if (erosion_ps->getAttribute("algorithm", d_erosionAlgorithm)) {
      if (d_erosionAlgorithm == "none") d_doErosion = false;
      else d_doErosion = true;
    }
  }

  dbg << "---------------------------------------------------------\n";
  dbg << "MPM Flags " << endl;
  dbg << "---------------------------------------------------------\n";
  dbg << " Nodes for interpolation     = " << d_8or27 << endl;
  dbg << " With Color                  = " << d_with_color << endl;
  dbg << " Artificial Damping Coeff    = " << d_artificialDampCoeff << endl;
  dbg << " Artificial Viscosity On     = " << d_artificial_viscosity<< endl;
  dbg << " Artificial Viscosity Coeff1 = " << d_artificialViscCoeff1<< endl;
  dbg << " Artificial Viscosity Coeff2 = " << d_artificialViscCoeff2<< endl;
  dbg << " Accumulate Strain Energy    = " << d_accStrainEnergy << endl;
  dbg << " Adiabatic Heating On        = " << d_adiabaticHeating << endl;
  dbg << " Create New Particles        = " << d_createNewParticles << endl;
  dbg << " Add New Material            = " << d_addNewMaterial << endl;
  dbg << " Do Erosion ?                = " << d_doErosion << endl;
  dbg << "  Erosion Algorithm          = " << d_erosionAlgorithm << endl;
  dbg << " Use Load Curves             = " << d_useLoadCurves << endl;
  dbg << " ForceBC increment factor    = " << d_forceIncrementFactor<< endl;
  dbg << "---------------------------------------------------------\n";
}

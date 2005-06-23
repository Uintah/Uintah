#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>
#include <Packages/Uintah/Core/Grid/Node27Interpolator.h>
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
  d_ref_temp = 0.0; // for thermal stress
  d_integrator_type = "explicit";
  d_integrator = Explicit;

  d_artificial_viscosity = false;
  d_artificialViscCoeff1 = 0.2;
  d_artificialViscCoeff2 = 2.0;
  d_accStrainEnergy = false;
  d_useLoadCurves = false;
  d_createNewParticles = false;
  d_addNewMaterial = false;
  d_with_color = false;
  d_fracture = false;
  d_minGridLevel = 0;
  d_maxGridLevel = 1000;
                      
  d_doErosion = false;
  d_erosionAlgorithm = "none";

  d_adiabaticHeating = 1.0;
  d_artificialDampCoeff = 0.0;
  d_forceIncrementFactor = 1.0;
  d_canAddMPMMaterial = false;
  d_interpolator = scinew LinearInterpolator(); 
  d_addFrictionWork = 1.0;  // do frictional heating by default

  d_extraSolverFlushes = 0;  // Have PETSc do more flushes to save memory
  d_doImplicitHeatConduction = false;
  d_doTransientImplicitHeatConduction = true;
}

MPMFlags::~MPMFlags()
{
  delete d_interpolator;
}

void
MPMFlags::readMPMFlags(ProblemSpecP& ps, const GridP& grid)
{
  ps->get("time_integrator", d_integrator_type);
  if (d_integrator_type == "implicit") 
    d_integrator = Implicit;
  else if (d_integrator_type == "fracture") {
    d_integrator = Fracture;
    d_fracture = true;
  }
  else 
    d_integrator = Explicit;
  ps->get("nodes8or27", d_8or27);
  ps->get("reference_temperature", d_ref_temp); // for thermal stress
  ps->get("withColor",  d_with_color);
  ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  ps->get("artificial_viscosity",     d_artificial_viscosity);
  ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  ps->get("accumulate_strain_energy", d_accStrainEnergy);
  ps->get("use_load_curves", d_useLoadCurves);
  bool adiabaticHeatingOn = false;
  ps->get("turn_on_adiabatic_heating", adiabaticHeatingOn);
  if (adiabaticHeatingOn) d_adiabaticHeating = 0.0;
  ps->getWithDefault("min_grid_level", d_minGridLevel, 0);
  ps->getWithDefault("max_grid_level", d_maxGridLevel, 1000);
  ps->get("ForceBC_force_increment_factor", d_forceIncrementFactor);
  ps->get("create_new_particles", d_createNewParticles);
  ps->get("manual_new_material", d_addNewMaterial);
  ps->get("CanAddMPMMaterial", d_canAddMPMMaterial);
  ps->get("DoImplicitHeatConduction", d_doImplicitHeatConduction);
  ps->get("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  bool do_contact_friction = true;
  ps->get("do_contact_friction_heating", do_contact_friction);
  if (!do_contact_friction) d_addFrictionWork = 0.0;
  ProblemSpecP erosion_ps = ps->findBlock("erosion");
  if (erosion_ps) {
    if (erosion_ps->getAttribute("algorithm", d_erosionAlgorithm)) {
      if (d_erosionAlgorithm == "none") d_doErosion = false;
      else d_doErosion = true;
    }
  }

  delete d_interpolator;

  if(d_8or27==8){
    d_interpolator = scinew LinearInterpolator();
  } else if(d_8or27==27){
    d_interpolator = scinew Node27Interpolator();
  }

  ps->get("extra_solver_flushes", d_extraSolverFlushes);

  if (dbg.active()) {
    dbg << "---------------------------------------------------------\n";
    dbg << "MPM Flags " << endl;
    dbg << "---------------------------------------------------------\n";
    dbg << " Time Integration            = " << d_integrator_type << endl;
    dbg << " Nodes for interpolation     = " << d_8or27 << endl;
    dbg << " Reference temperature       = " << d_ref_temp << endl;  
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
    dbg << " Contact Friction Heating    = " << d_addFrictionWork << endl;
    dbg << " Extra Solver flushes        = " << d_extraSolverFlushes << endl;
    dbg << "---------------------------------------------------------\n";
  }
}

bool MPMFlags::doMPMOnLevel(int level) const
{
  return level >= d_minGridLevel && level <= d_maxGridLevel;
}

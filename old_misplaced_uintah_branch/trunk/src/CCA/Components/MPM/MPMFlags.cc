#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/Node27Interpolator.h>
#include <Core/Grid/TOBSplineInterpolator.h>
#include <Core/Grid/BSplineInterpolator.h>
//#include <Core/Grid/AMRInterpolator.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg("MPMFlags", false);

MPMFlags::MPMFlags()
{
  d_interpolator_type = "linear";
  d_ref_temp = 0.0; // for thermal stress
  d_integrator_type = "explicit";
  d_integrator = Explicit;
  d_AMR = false;

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
  d_doThermalExpansion = true;

  d_adiabaticHeatingOn = false;
  d_adiabaticHeating = 1.0;
  d_artificialDampCoeff = 0.0;
  d_forceIncrementFactor = 1.0;
  d_canAddMPMMaterial = false;
  d_interpolator = scinew LinearInterpolator(); 
  d_do_contact_friction = true;
  d_addFrictionWork = 1.0;  // don't do frictional heating by default

  d_extraSolverFlushes = 0;  // Have PETSc do more flushes to save memory
  d_doImplicitHeatConduction = false;
  d_doExplicitHeatConduction = true;
  d_doTransientImplicitHeatConduction = true;
  d_doGridReset = true;
  d_min_part_mass = 3.e-15;
  d_max_vel = 3.e105;
  d_with_ice = false;
}

MPMFlags::~MPMFlags()
{
  delete d_interpolator;
}

void
MPMFlags::readMPMFlags(ProblemSpecP& ps)
{
  ProblemSpecP root = ps->getRootNode();
  ProblemSpecP mpm_flag_ps = root->findBlock("MPM");

  if (!mpm_flag_ps)
    return;

  mpm_flag_ps->get("time_integrator", d_integrator_type);
  if (d_integrator_type == "implicit") 
    d_integrator = Implicit;
  else if (d_integrator_type == "fracture") {
    d_integrator = Fracture;
    d_fracture = true;
  }
  else{
    d_integrator = Explicit;
  }
  int junk=0;
  mpm_flag_ps->get("nodes8or27", junk);
  if(junk!=0){
     cerr << "nodes8or27 is deprecated, use " << endl;
     cerr << "<interpolator>type</interpolator>" << endl;
     cerr << "where type is one of the following:" << endl;
     cerr << "linear, gimp, 4thorderBS" << endl;
    exit(1);
  }

  mpm_flag_ps->get("interpolator", d_interpolator_type);
  mpm_flag_ps->get("AMR", d_AMR);
  mpm_flag_ps->get("reference_temperature", d_ref_temp); // for thermal stress
  mpm_flag_ps->get("withColor",  d_with_color);
  mpm_flag_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  mpm_flag_ps->get("artificial_viscosity",     d_artificial_viscosity);
  mpm_flag_ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  mpm_flag_ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  mpm_flag_ps->get("accumulate_strain_energy", d_accStrainEnergy);
  mpm_flag_ps->get("use_load_curves", d_useLoadCurves);

  mpm_flag_ps->get("turn_on_adiabatic_heating", d_adiabaticHeatingOn);
  if (d_adiabaticHeatingOn) d_adiabaticHeating = 0.0;
  mpm_flag_ps->get("ForceBC_force_increment_factor", d_forceIncrementFactor);
  mpm_flag_ps->get("create_new_particles", d_createNewParticles);
  mpm_flag_ps->get("manual_new_material", d_addNewMaterial);
  mpm_flag_ps->get("CanAddMPMMaterial", d_canAddMPMMaterial);
  mpm_flag_ps->get("DoImplicitHeatConduction", d_doImplicitHeatConduction);
  mpm_flag_ps->get("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  mpm_flag_ps->get("DoExplicitHeatConduction", d_doExplicitHeatConduction);
  mpm_flag_ps->get("DoThermalExpansion", d_doThermalExpansion);
  mpm_flag_ps->get("do_grid_reset",      d_doGridReset);
  mpm_flag_ps->get("minimum_particle_mass",    d_min_part_mass);
  mpm_flag_ps->get("maximum_particle_velocity",d_max_vel);

  mpm_flag_ps->get("do_contact_friction_heating", d_do_contact_friction);
  if (!d_do_contact_friction) d_addFrictionWork = 0.0;
  ProblemSpecP erosion_ps = mpm_flag_ps->findBlock("erosion");
  if (erosion_ps) {
    if (erosion_ps->getAttribute("algorithm", d_erosionAlgorithm)) {
      if (d_erosionAlgorithm == "none") d_doErosion = false;
      else d_doErosion = true;
    }
  }

  delete d_interpolator;

  if(d_interpolator_type=="linear"){
    d_interpolator = scinew LinearInterpolator();
    d_8or27 = 8;
  } else if(d_interpolator_type=="gimp"){
    d_interpolator = scinew Node27Interpolator();
    d_8or27 = 27;
  } else if(d_interpolator_type=="3rdorderBS"){
    d_interpolator = scinew TOBSplineInterpolator();
    d_8or27 = 27;
  } else if(d_interpolator_type=="4thorderBS"){
    d_interpolator = scinew BSplineInterpolator();
    d_8or27 = 64;
  }

  mpm_flag_ps->get("extra_solver_flushes", d_extraSolverFlushes);

  mpm_flag_ps->get("boundary_traction_faces", d_bndy_face_txt_list);

  if (dbg.active()) {
    dbg << "---------------------------------------------------------\n";
    dbg << "MPM Flags " << endl;
    dbg << "---------------------------------------------------------\n";
    dbg << " Time Integration            = " << d_integrator_type << endl;
    dbg << " Interpolation type          = " << d_interpolator_type << endl;
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

void
MPMFlags::outputProblemSpec(ProblemSpecP& ps)
{

  ps->appendElement("time_integrator", d_integrator_type);

  ps->appendElement("interpolator", d_interpolator_type);
  ps->appendElement("reference_temperature", d_ref_temp);
  ps->appendElement("withColor",  d_with_color);
  ps->appendElement("artificial_damping_coeff", d_artificialDampCoeff);
  ps->appendElement("artificial_viscosity",     d_artificial_viscosity);
  ps->appendElement("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  ps->appendElement("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  ps->appendElement("accumulate_strain_energy", d_accStrainEnergy);
  ps->appendElement("use_load_curves", d_useLoadCurves);
  ps->appendElement("turn_on_adiabatic_heating", d_adiabaticHeatingOn);
  ps->appendElement("ForceBC_force_increment_factor", d_forceIncrementFactor);
  ps->appendElement("create_new_particles", d_createNewParticles);
  ps->appendElement("manual_new_material", d_addNewMaterial);
  ps->appendElement("CanAddMPMMaterial", d_canAddMPMMaterial);
  ps->appendElement("DoImplicitHeatConduction", d_doImplicitHeatConduction);
  ps->appendElement("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  ps->appendElement("DoExplicitHeatConduction", d_doExplicitHeatConduction);
  ps->appendElement("DoThermalExpansion", d_doThermalExpansion);
  ps->appendElement("do_grid_reset",      d_doGridReset);
  ps->appendElement("minimum_particle_mass",    d_min_part_mass);
  ps->appendElement("maximum_particle_velocity",d_max_vel);

  ps->appendElement("do_contact_friction_heating", d_do_contact_friction);

  ProblemSpecP erosion_ps = ps->appendChild("erosion");
  erosion_ps->setAttribute("algorithm", d_erosionAlgorithm);

  ps->appendElement("extra_solver_flushes", d_extraSolverFlushes);

  ps->appendElement("boundary_traction_faces", d_bndy_face_txt_list);
}


bool
MPMFlags::doMPMOnLevel(int level, int numLevels) const
{
  return (level >= d_minGridLevel && level <= d_maxGridLevel) ||
          (d_minGridLevel < 0 && level == numLevels + d_minGridLevel);
}

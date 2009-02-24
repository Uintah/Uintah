/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>
#include <Packages/Uintah/Core/Grid/Node27Interpolator.h>
#include <Packages/Uintah/Core/Grid/TOBSplineInterpolator.h>
#include <Packages/Uintah/Core/Grid/BSplineInterpolator.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg("MPMFlags", false);

MPMFlags::MPMFlags(const ProcessorGroup* myworld)
{
  d_interpolator_type = "linear";
  d_ref_temp = 0.0; // for thermal stress
  d_integrator_type = "explicit";
  d_integrator = Explicit;
  d_AMR = false;
  d_axisymmetric = false;

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
  d_computeNodalHeatFlux = false;
  d_doTransientImplicitHeatConduction = true;
  d_doGridReset = true;
  d_min_part_mass = 3.e-15;
  d_max_vel = 3.e105;
  d_with_ice = false;
  d_with_arches = false;
  d_myworld = myworld;
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
     cerr << "linear, gimp, 3rdorderBS" << endl;
    exit(1);
  }

  mpm_flag_ps->get("interpolator", d_interpolator_type);
  mpm_flag_ps->get("AMR", d_AMR);
  mpm_flag_ps->get("axisymmetric", d_axisymmetric);
  mpm_flag_ps->get("reference_temperature", d_ref_temp); // for thermal stress
  mpm_flag_ps->get("withColor",  d_with_color);
  mpm_flag_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  mpm_flag_ps->get("artificial_viscosity",     d_artificial_viscosity);
  mpm_flag_ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  mpm_flag_ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  mpm_flag_ps->get("accumulate_strain_energy", d_accStrainEnergy);
  mpm_flag_ps->get("use_load_curves", d_useLoadCurves);

  if(d_artificial_viscosity && d_integrator_type == "implicit"){
    if (d_myworld->myrank() == 0){
      cerr << "artificial viscosity is not implemented" << endl;
      cerr << "with implicit time integration" << endl;
    }
  }

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
  
  // d_doComputeHeatFlux:
  // set to true if the label g.HeatFlux is saved or 
  // flatPlat_heatFlux analysis module is used.
  //
  // orginal problem spec
  ProblemSpecP DA_ps = root->findBlock("DataArchiver");
  if(DA_ps){
    for(ProblemSpecP label_iter = DA_ps->findBlock("save"); label_iter != 0;
                     label_iter = label_iter->findNextBlock("save")){
      map<string,string> labelName;

      label_iter->getAttributes(labelName);
      if(labelName["label"] == "g.HeatFlux"){
        d_computeNodalHeatFlux = true;
      }
    }
  }

  ProblemSpecP da_ps = root->findBlock("DataAnalysis");

  if (da_ps) {
    ProblemSpecP module_ps = da_ps->findBlock("Module");
    if(module_ps){
      map<string,string> attributes;
      module_ps->getAttributes(attributes);
      if ( attributes["name"]== "flatPlate_heatFlux") {
        d_computeNodalHeatFlux = true;
      }
    }
  }
  // restart problem spec
  mpm_flag_ps->get("computeNodalHeatFlux",d_computeNodalHeatFlux);
  
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
  } else{
    ostringstream warn;
    warn << "ERROR:MPM: invalid interpolation type ("<<d_interpolator_type << ")"
         << "Valid options are: \n"
         << "linear\n"
         << "gimp\n"
         << "3rdorderBS\n"
         << "4thorderBS\n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
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
  ps->appendElement("AMR", d_AMR);
  ps->appendElement("axisymmetric", d_axisymmetric);
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
  ps->appendElement("computeNodalHeatFlux",d_computeNodalHeatFlux);
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

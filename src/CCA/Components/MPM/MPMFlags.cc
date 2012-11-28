/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/GIMPInterpolator.h>
#include <Core/Grid/AxiGIMPInterpolator.h>
#include <Core/Grid/cpdiInterpolator.h>
#include <Core/Grid/axiCpdiInterpolator.h>
#include <Core/Grid/fastCpdiInterpolator.h>
#include <Core/Grid/fastAxiCpdiInterpolator.h>
#include <Core/Grid/TOBSplineInterpolator.h>
#include <Core/Grid/BSplineInterpolator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <iostream>

using namespace Uintah;
using namespace std;

static DebugStream dbg("MPMFlags", false);

MPMFlags::MPMFlags(const ProcessorGroup* myworld)
{
  d_gravity = Vector(0.,0.,0.);
  d_interpolator_type = "linear";
  d_integrator_type = "explicit";
  d_integrator = Explicit;
  d_AMR = false;
  d_axisymmetric = false;

  d_artificial_viscosity = false;
  d_artificial_viscosity_heating = false;
  d_artificialViscCoeff1 = 0.2;
  d_artificialViscCoeff2 = 2.0;
  d_useLoadCurves = false;
  d_useCBDI = false;
  d_useCohesiveZones = false;
  d_createNewParticles = false;
  d_addNewMaterial = false;
  d_with_color = false;
  d_fracture = false;
  d_minGridLevel = 0;
  d_maxGridLevel = 1000;
                      
  d_erosionAlgorithm = "none";
  d_doErosion = false;
  d_deleteRogueParticles = false;
  d_doThermalExpansion = true;

  d_artificialDampCoeff = 0.0;
  d_forceIncrementFactor = 1.0;
  d_canAddMPMMaterial = false;
  d_interpolator = scinew LinearInterpolator(); 
  d_do_contact_friction = false;
  d_addFrictionWork = 0.0;  // don't do frictional heating by default

  d_extraSolverFlushes = 0;  // Have PETSc do more flushes to save memory
  d_doImplicitHeatConduction = false;
  d_doExplicitHeatConduction = true;
  d_doPressureStabilization = false;
  d_computeNodalHeatFlux = false;
  d_computeScaleFactor = false;
  d_doTransientImplicitHeatConduction = true;
  d_prescribeDeformation = false;
  d_prescribedDeformationFile = "time_defgrad_rotation";
  d_exactDeformation = false;
  d_insertParticles = false;
  d_doGridReset = true;
  d_min_part_mass = 3.e-15;
  d_min_mass_for_acceleration = 0;// Min mass to allow division by in computing acceleration
  d_max_vel = 3.e105;
  d_with_ice = false;
  d_with_arches = false;
  d_use_momentum_form = false;
  d_myworld = myworld;
  
  d_reductionVars = scinew reductionVars();
  d_reductionVars->mass             = false;
  d_reductionVars->momentum         = false;
  d_reductionVars->thermalEnergy    = false;
  d_reductionVars->strainEnergy     = false;
  d_reductionVars->accStrainEnergy  = false;
  d_reductionVars->KE               = false;
  d_reductionVars->volDeformed      = false;
  d_reductionVars->centerOfMass     = false;

// MMS
if(d_mms_type=="AxisAligned"){
    d_mms_type = "AxisAligned";
  } else if(d_mms_type=="GeneralizedVortex"){
    d_mms_type = "GeneralizedVortex";
  } else if(d_mms_type=="ExpandingRing"){
    d_mms_type = "ExpandingRing";
  } else if(d_mms_type=="AxisAligned3L"){
    d_mms_type = "AxisAligned3L";
  }
}

MPMFlags::~MPMFlags()
{
  delete d_interpolator;
  delete d_reductionVars;
}

void
MPMFlags::readMPMFlags(ProblemSpecP& ps, Output* dataArchive)
{
  ProblemSpecP root = ps->getRootNode();
  ProblemSpecP mpm_flag_ps = root->findBlock("MPM");
  ProblemSpecP phys_cons_ps = root->findBlock("PhysicalConstants");

  if(phys_cons_ps){
    phys_cons_ps->require("gravity",d_gravity);
  } else if (mpm_flag_ps) {
    mpm_flag_ps->require("gravity",d_gravity);
  } else{
    d_gravity=Vector(0,0,0);
  }

  //__________________________________
  //  Set the on/off flags to determine which
  // reduction variables are computed
  d_reductionVars->mass           = dataArchive->isLabelSaved( "TotalMass" );
  d_reductionVars->momentum       = dataArchive->isLabelSaved( "TotalMomentum" );
  d_reductionVars->thermalEnergy  = dataArchive->isLabelSaved( "ThermalEnergy" );
  d_reductionVars->KE             = dataArchive->isLabelSaved( "KineticEnergy" );
  d_reductionVars->strainEnergy   = dataArchive->isLabelSaved( "StrainEnergy" );
  d_reductionVars->accStrainEnergy= dataArchive->isLabelSaved( "AccStrainEnergy" );
  d_reductionVars->volDeformed    = dataArchive->isLabelSaved( "TotalVolumeDeformed" );
  d_reductionVars->centerOfMass   = dataArchive->isLabelSaved( "CenterOfMassPosition" );
 
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
     cerr << "linear, gimp, cpgimp, 3rdorderBS, cpdi, fastcpdi" << endl;
    exit(1);
  }

  mpm_flag_ps->get("interpolator", d_interpolator_type);
  mpm_flag_ps->get("AMR", d_AMR);
  mpm_flag_ps->get("axisymmetric", d_axisymmetric);
  mpm_flag_ps->get("withColor",  d_with_color);
  mpm_flag_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  mpm_flag_ps->get("artificial_viscosity",     d_artificial_viscosity);
  if(d_artificial_viscosity){
    d_artificial_viscosity_heating=true;
  }
  mpm_flag_ps->get("artificial_viscosity_heating",d_artificial_viscosity_heating);
  mpm_flag_ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  mpm_flag_ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  mpm_flag_ps->get("use_load_curves", d_useLoadCurves);
  mpm_flag_ps->get("use_CBDI_boundary_condition", d_useCBDI);
  mpm_flag_ps->get("exactDeformation",d_exactDeformation);
  mpm_flag_ps->get("use_cohesive_zones", d_useCohesiveZones);

  if(d_artificial_viscosity && d_integrator_type == "implicit"){
    if (d_myworld->myrank() == 0){
      cerr << "artificial viscosity is not implemented" << endl;
      cerr << "with implicit time integration" << endl;
    }
  }

  if(!d_artificial_viscosity && d_artificial_viscosity_heating){
    ostringstream warn;
    warn << "ERROR:MPM: You can't have heating due to artificial viscosity "
         << "if artificial_viscosity is not enabled." << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
  }

  mpm_flag_ps->get("ForceBC_force_increment_factor", d_forceIncrementFactor);
  mpm_flag_ps->get("create_new_particles", d_createNewParticles);
  mpm_flag_ps->get("manual_new_material", d_addNewMaterial);
  mpm_flag_ps->get("CanAddMPMMaterial", d_canAddMPMMaterial);
  mpm_flag_ps->get("DoImplicitHeatConduction", d_doImplicitHeatConduction);
  mpm_flag_ps->get("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  mpm_flag_ps->get("DoExplicitHeatConduction", d_doExplicitHeatConduction);
  mpm_flag_ps->get("DoPressureStabilization", d_doPressureStabilization);
  mpm_flag_ps->get("DoThermalExpansion", d_doThermalExpansion);
  mpm_flag_ps->get("do_grid_reset",      d_doGridReset);
  mpm_flag_ps->get("minimum_particle_mass",    d_min_part_mass);
  mpm_flag_ps->get("minimum_mass_for_acc",     d_min_mass_for_acceleration);
  mpm_flag_ps->get("maximum_particle_velocity",d_max_vel);
  mpm_flag_ps->get("UsePrescribedDeformation",d_prescribeDeformation);
  if(d_prescribeDeformation){
    mpm_flag_ps->get("PrescribedDeformationFile",d_prescribedDeformationFile);
  }
//MMS
  mpm_flag_ps->get("RunMMSProblem",d_mms_type);

  mpm_flag_ps->get("InsertParticles",d_insertParticles);
  if(d_insertParticles){
    mpm_flag_ps->require("InsertParticlesFile",d_insertParticlesFile);
  }

  mpm_flag_ps->get("do_contact_friction_heating", d_do_contact_friction);
  if (!d_do_contact_friction) d_addFrictionWork = 0.0;

   ProblemSpecP erosion_ps = mpm_flag_ps->findBlock("erosion");
   if (erosion_ps) {
     if (erosion_ps->getAttribute("algorithm", d_erosionAlgorithm)) {
       if (d_erosionAlgorithm == "none") d_doErosion = false;
       else d_doErosion = true;
     }
   }

  mpm_flag_ps->get("delete_rogue_particles",  d_deleteRogueParticles);
  
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
      if(labelName["label"] == "p.scalefactor"){
        d_computeScaleFactor = true;
      }
    }
  }

  ProblemSpecP da_ps = root->findBlock("DataAnalysis");

  if (da_ps) {
    for(ProblemSpecP module_ps = da_ps->findBlock("Module"); module_ps != 0;
                     module_ps = module_ps->findNextBlock("Module")){
      if(module_ps){
        map<string,string> attributes;
        module_ps->getAttributes(attributes);
        if ( attributes["name"]== "flatPlate_heatFlux") {
          d_computeNodalHeatFlux = true;
        }
      }
    }
  }
  // restart problem spec
  mpm_flag_ps->get("computeNodalHeatFlux",d_computeNodalHeatFlux);
  mpm_flag_ps->get("computeScaleFactor",  d_computeScaleFactor);
  
  delete d_interpolator;

  if(d_interpolator_type=="linear"){
    d_interpolator = scinew LinearInterpolator();
  } else if(d_interpolator_type=="gimp"){
    if(d_axisymmetric){
      d_interpolator = scinew AxiGIMPInterpolator();
    } else{
      d_interpolator = scinew GIMPInterpolator();
    }
  } else if(d_interpolator_type=="cpgimp"){
    d_interpolator = scinew GIMPInterpolator();
  } else if(d_interpolator_type=="3rdorderBS"){
    d_interpolator = scinew TOBSplineInterpolator();
  } else if(d_interpolator_type=="4thorderBS"){
    d_interpolator = scinew BSplineInterpolator();
  } else if(d_interpolator_type=="cpdi"){
    if(d_axisymmetric){
      d_interpolator = scinew axiCpdiInterpolator();
    } else{
      d_interpolator = scinew cpdiInterpolator();
    }
  } else if(d_interpolator_type=="fastcpdi"){
    if(d_axisymmetric){
      d_interpolator = scinew fastAxiCpdiInterpolator();
    } else{
      d_interpolator = scinew fastCpdiInterpolator();
    }
  }else{
    ostringstream warn;
    warn << "ERROR:MPM: invalid interpolation type ("<<d_interpolator_type << ")"
         << "Valid options are: \n"
         << "linear\n"
         << "gimp\n"
         << "cpgimp\n"
         << "cpdi\n"
         << "3rdorderBS\n"
         << "4thorderBS\n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
  }
  // Get the size of the vectors associated with the interpolator
  d_8or27=interpolator->size();

  mpm_flag_ps->get("extra_solver_flushes", d_extraSolverFlushes);

  mpm_flag_ps->get("boundary_traction_faces", d_bndy_face_txt_list);

  mpm_flag_ps->get("UseMomentumForm", d_use_momentum_form);

  if (dbg.active()) {
    dbg << "---------------------------------------------------------\n";
    dbg << "MPM Flags " << endl;
    dbg << "---------------------------------------------------------\n";
    dbg << " Time Integration            = " << d_integrator_type << endl;
    dbg << " Interpolation type          = " << d_interpolator_type << endl;
    dbg << " With Color                  = " << d_with_color << endl;
    dbg << " Artificial Damping Coeff    = " << d_artificialDampCoeff << endl;
    dbg << " Artificial Viscosity On     = " << d_artificial_viscosity<< endl;
    dbg << " Artificial Viscosity Htng   = " << d_artificial_viscosity_heating<< endl;
    dbg << " Artificial Viscosity Coeff1 = " << d_artificialViscCoeff1<< endl;
    dbg << " Artificial Viscosity Coeff2 = " << d_artificialViscCoeff2<< endl;
    dbg << " Create New Particles        = " << d_createNewParticles << endl;
    dbg << " Add New Material            = " << d_addNewMaterial << endl;
    dbg << " Delete Rogue Particles?     = " << d_deleteRogueParticles << endl;
    dbg << " Use Load Curves             = " << d_useLoadCurves << endl;
    dbg << " Use CBDI boundary condition = " << d_useCBDI << endl;
    dbg << " Use Cohesive Zones          = " << d_useCohesiveZones << endl;
    dbg << " ForceBC increment factor    = " << d_forceIncrementFactor<< endl;
    dbg << " Contact Friction Heating    = " << d_addFrictionWork << endl;
    dbg << " Extra Solver flushes        = " << d_extraSolverFlushes << endl;
    dbg << "---------------------------------------------------------\n";
  }
}

void
MPMFlags::outputProblemSpec(ProblemSpecP& ps)
{

  ps->appendElement("gravity", d_gravity);
  ps->appendElement("time_integrator", d_integrator_type);

  ps->appendElement("interpolator", d_interpolator_type);
  ps->appendElement("AMR", d_AMR);
  ps->appendElement("axisymmetric", d_axisymmetric);
  ps->appendElement("withColor",  d_with_color);
  ps->appendElement("artificial_damping_coeff", d_artificialDampCoeff);
  ps->appendElement("artificial_viscosity",     d_artificial_viscosity);
  ps->appendElement("artificial_viscosity_heating", d_artificial_viscosity_heating);
  ps->appendElement("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  ps->appendElement("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  ps->appendElement("use_cohesive_zones", d_useCohesiveZones);
  ps->appendElement("use_load_curves", d_useLoadCurves);
  ps->appendElement("use_CBDI_boundary_condition", d_useCBDI);
  ps->appendElement("exactDeformation",d_exactDeformation);
  ps->appendElement("ForceBC_force_increment_factor", d_forceIncrementFactor);
  ps->appendElement("create_new_particles", d_createNewParticles);
  ps->appendElement("manual_new_material", d_addNewMaterial);
  ps->appendElement("CanAddMPMMaterial", d_canAddMPMMaterial);
  ps->appendElement("DoImplicitHeatConduction", d_doImplicitHeatConduction);
  ps->appendElement("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  ps->appendElement("DoExplicitHeatConduction", d_doExplicitHeatConduction);
  ps->appendElement("DoPressureStabilization", d_doPressureStabilization);
  ps->appendElement("computeNodalHeatFlux",d_computeNodalHeatFlux);
  ps->appendElement("computeScaleFactor",  d_computeScaleFactor);
  ps->appendElement("DoThermalExpansion", d_doThermalExpansion);
  ps->appendElement("do_grid_reset",      d_doGridReset);
  ps->appendElement("minimum_particle_mass",    d_min_part_mass);
  ps->appendElement("minimum_mass_for_acc",     d_min_mass_for_acceleration);
  ps->appendElement("maximum_particle_velocity",d_max_vel);
  ps->appendElement("UsePrescribedDeformation",d_prescribeDeformation);
  if(d_prescribeDeformation){
    ps->appendElement("PrescribedDeformationFile",d_prescribedDeformationFile);
  }
//MMS
  ps->appendElement("RunMMSProblem",d_mms_type);
  ps->appendElement("InsertParticles",d_insertParticles);
  if(d_insertParticles){
    ps->appendElement("InsertParticlesFile",d_insertParticlesFile);
  }

  ps->appendElement("do_contact_friction_heating", d_do_contact_friction);

  ps->appendElement("delete_rogue_particles",d_deleteRogueParticles);

  ProblemSpecP erosion_ps = ps->appendChild("erosion");
  erosion_ps->setAttribute("algorithm", d_erosionAlgorithm);
 
  ps->appendElement("extra_solver_flushes", d_extraSolverFlushes);

  ps->appendElement("boundary_traction_faces", d_bndy_face_txt_list);

  ps->appendElement("UseMomentumForm", d_use_momentum_form);
}


bool
MPMFlags::doMPMOnLevel(int level, int numLevels) const
{
  return (level >= d_minGridLevel && level <= d_maxGridLevel) ||
          (d_minGridLevel < 0 && level == numLevels + d_minGridLevel);
}

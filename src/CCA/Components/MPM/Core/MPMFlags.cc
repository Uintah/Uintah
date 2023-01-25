/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Ports/Output.h>

#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/AxiLinearInterpolator.h>
#include <Core/Grid/GIMPInterpolator.h>
#include <Core/Grid/AxiGIMPInterpolator.h>
#include <Core/Grid/cpdiInterpolator.h>
#include <Core/Grid/fastCpdiInterpolator.h>
#include <Core/Grid/axiCpdiInterpolator.h>
#include <Core/Grid/cptiInterpolator.h>
#include <Core/Grid/axiCptiInterpolator.h>
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
  d_particle_ghost_type           = Ghost::None;
  d_particle_ghost_layer          = 0;

  d_gravity                       =  Vector(0.,0.,0.);
  d_interpolator_type             =  "linear";
  d_integrator_type               =  "explicit";
  d_integrator                    =  Explicit;
  d_AMR                           =  false;
  d_SingleFieldMPM                =  false;
  d_axisymmetric                  =  false;
  d_artificial_viscosity          =  false;
  d_artificial_viscosity_heating  =  false;
  d_artificialViscCoeff1          =  0.2;
  d_artificialViscCoeff2          =  2.0;
  d_useLoadCurves                 =  false;
  d_useCBDI                       =  false;
  d_useCPTI                       =  false;
  d_useCohesiveZones              =  false;
  d_with_color                    =  false;
  d_fracture                      =  false;
  d_minGridLevel                  =  0;
  d_maxGridLevel                  =  1000;
  d_doThermalExpansion            =  true;
  d_refineParticles               =  false;
  d_XPIC2                         =  false;
  d_artificialDampCoeff           =  0.0;
  d_interpolator                  =  scinew LinearInterpolator();
  d_do_contact_friction           =  false;
  d_computeNormals                =  false;
  d_useLogisticRegression         =  false;
  d_computeColinearNormals        =  true;
  d_restartOnLargeNodalVelocity   =  false;
  d_ndim                          =  3;
  d_addFrictionWork               =  0.0;               // don't do frictional heating by default

  d_extraSolverFlushes                 =  0;            // Have PETSc do more flushes to save memory
  d_doImplicitHeatConduction           =  false;
  d_doExplicitHeatConduction           =  true;
  d_deleteGeometryObjects              =  false;
  d_doPressureStabilization            =  false;
  d_computeNodalHeatFlux               =  false;
  d_computeScaleFactor                 =  false;
  d_doTransientImplicitHeatConduction  =  true;
  d_prescribeDeformation               =  false;
  d_prescribedDeformationFile          =  "time_defgrad_rotation";
  d_exactDeformation                   =  false;
  d_insertParticles                    =  false;
  d_doGridReset                        =  true;
  d_min_part_mass                      =  3.e-15;
  d_min_subcycles_for_F                =  1;
  d_min_mass_for_acceleration          =  0;            // Min mass to allow division by in computing acceleration
  d_max_vel                            =  3.e105;
  d_with_ice                           =  false;
  d_with_arches                        =  false;
  d_myworld                            =  myworld;
  
  d_reductionVars = scinew reductionVars();
  d_reductionVars->mass             = false;
  d_reductionVars->momentum         = false;
  d_reductionVars->thermalEnergy    = false;
  d_reductionVars->strainEnergy     = false;
  d_reductionVars->accStrainEnergy  = false;
  d_reductionVars->KE               = false;
  d_reductionVars->volDeformed      = false;
  d_reductionVars->centerOfMass     = false;

  //******* Reactive Flow Component
  d_doScalarDiffusion   =  false;  // for diffusion component found  in ReactiveFlow
  d_doAutoCycleBC       =  false;  // for scalar flux boundary conditions
  d_autoCycleUseMinMax  =  false;
  d_autoCycleMax        =  .9;
  d_autoCycleMin        =  .1;
  d_withGaussSolver     =  false;

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
  d_reductionVars->mass           = dataArchive->isLabelSaved("TotalMass");
  d_reductionVars->momentum       = dataArchive->isLabelSaved("TotalMomentum");
  d_reductionVars->thermalEnergy  = dataArchive->isLabelSaved("ThermalEnergy");
  d_reductionVars->KE             = dataArchive->isLabelSaved("KineticEnergy");
  d_reductionVars->strainEnergy   = dataArchive->isLabelSaved("StrainEnergy");
  d_reductionVars->accStrainEnergy= dataArchive->isLabelSaved("AccStrainEnergy");
  d_reductionVars->volDeformed    = dataArchive->isLabelSaved("TotalVolumeDeformed");
  d_reductionVars->centerOfMass   = dataArchive->isLabelSaved("CenterOfMassPosition");
 
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
     cerr << "linear, gimp, 3rdorderBS, cpdi, cpti" << endl;
    exit(1);
  }

  mpm_flag_ps->get("interpolator", d_interpolator_type);
  mpm_flag_ps->getWithDefault("cpdi_lcrit", d_cpdi_lcrit, 1.e10);
  mpm_flag_ps->get("AMR", d_AMR);
  mpm_flag_ps->get("axisymmetric", d_axisymmetric);
  mpm_flag_ps->get("withColor",  d_with_color);
  mpm_flag_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
  mpm_flag_ps->get("artificial_viscosity",     d_artificial_viscosity);
  mpm_flag_ps->get("refine_particles",         d_refineParticles);
  mpm_flag_ps->get("XPIC2",                    d_XPIC2);
  if(d_artificial_viscosity){
    d_artificial_viscosity_heating=true;
  }
  mpm_flag_ps->get("artificial_viscosity_heating",d_artificial_viscosity_heating);
  mpm_flag_ps->get("artificial_viscosity_coeff1", d_artificialViscCoeff1);
  mpm_flag_ps->get("artificial_viscosity_coeff2", d_artificialViscCoeff2);
  mpm_flag_ps->get("use_load_curves",             d_useLoadCurves);
  mpm_flag_ps->get("use_CBDI_boundary_condition", d_useCBDI);
  mpm_flag_ps->get("exactDeformation",            d_exactDeformation);
  mpm_flag_ps->get("use_cohesive_zones",          d_useCohesiveZones);

  if(d_artificial_viscosity && d_integrator_type == "implicit"){
    if (d_myworld->myRank() == 0){
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

  mpm_flag_ps->get("DoImplicitHeatConduction",          d_doImplicitHeatConduction);
  mpm_flag_ps->get("DoTransientImplicitHeatConduction", d_doTransientImplicitHeatConduction);
  mpm_flag_ps->get("DoExplicitHeatConduction",          d_doExplicitHeatConduction);
  mpm_flag_ps->get("DeleteGeometryObjects",             d_deleteGeometryObjects);
  mpm_flag_ps->get("DoPressureStabilization",           d_doPressureStabilization);
  mpm_flag_ps->get("DoThermalExpansion",                d_doThermalExpansion);
  mpm_flag_ps->getWithDefault("UseGradientEnhancedVelocityProjection",  d_GEVelProj,false);
  mpm_flag_ps->get("do_grid_reset",                     d_doGridReset);
  mpm_flag_ps->get("minimum_particle_mass",             d_min_part_mass);
  mpm_flag_ps->get("minimum_subcycles_for_F",           d_min_subcycles_for_F);
  mpm_flag_ps->get("minimum_mass_for_acc",              d_min_mass_for_acceleration);
  mpm_flag_ps->get("maximum_particle_velocity",         d_max_vel);
  mpm_flag_ps->get("UsePrescribedDeformation",          d_prescribeDeformation);

  if(d_prescribeDeformation){
    mpm_flag_ps->get("PrescribedDeformationFile",d_prescribedDeformationFile);
  }
//MMS
  mpm_flag_ps->get("RunMMSProblem",d_mms_type);
  // Flag for CPTI interpolator
  if(d_interpolator_type=="cpti"){
    d_useCPTI = true;
  }
  mpm_flag_ps->get("InsertParticles",d_insertParticles);
  if(d_insertParticles){
    mpm_flag_ps->require("InsertParticlesFile",d_insertParticlesFile);
  }

  mpm_flag_ps->get("do_contact_friction_heating",d_do_contact_friction);
  mpm_flag_ps->get("computeNormals",             d_computeNormals);
  mpm_flag_ps->get("useLogisticRegression",       d_useLogisticRegression);
  mpm_flag_ps->get("computeColinearNormals",     d_computeColinearNormals);
  mpm_flag_ps->get("d_ndim",                      d_ndim);
  mpm_flag_ps->get("restartOnLargeNodalVelocity",d_restartOnLargeNodalVelocity);
  if (!d_do_contact_friction){
    d_addFrictionWork = 0.0;
  }

  // Setting Scalar Diffusion
  mpm_flag_ps->get("do_scalar_diffusion", d_doScalarDiffusion);
  mpm_flag_ps->get("do_auto_cycle_bc", d_doAutoCycleBC);
  mpm_flag_ps->get("auto_cycle_use_minmax", d_autoCycleUseMinMax);
  mpm_flag_ps->get("auto_cycle_max", d_autoCycleMax);
  mpm_flag_ps->get("auto_cycle_min", d_autoCycleMin);
  mpm_flag_ps->get("with_gauss_solver", d_withGaussSolver);
  
  
  d_computeScaleFactor = dataArchive->isLabelSaved("p.scalefactor");


  // d_doComputeHeatFlux if the label g.HeatFlux is saved or if
  // flatPlat_heatFlux analysis module is used.
  d_computeNodalHeatFlux   = dataArchive->isLabelSaved("g.HeatFlux");

  ProblemSpecP da_ps = root->findBlock("DataAnalysis");

  if (da_ps) {
    for( ProblemSpecP module_ps = da_ps->findBlock("Module"); module_ps != nullptr; module_ps = module_ps->findNextBlock( "Module" ) ) {
      if( module_ps ){
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
    if(d_axisymmetric){
      d_interpolator = scinew AxiLinearInterpolator();
    } else{
      d_interpolator = scinew LinearInterpolator();
    }
  } else if(d_interpolator_type=="gimp"){
    if(d_axisymmetric){
      d_interpolator = scinew AxiGIMPInterpolator();
    } else{
      d_interpolator = scinew GIMPInterpolator();
    }
  } else if(d_interpolator_type=="3rdorderBS"){
    if(!d_axisymmetric){
      d_interpolator = scinew TOBSplineInterpolator();
    } else{
      ostringstream warn;
      warn << "ERROR:MPM: invalid interpolation type ("<<d_interpolator_type << ")"
           << "Can't be used with axisymmetry at this time \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
    }
  } else if(d_interpolator_type=="4thorderBS"){
    if(!d_axisymmetric){
      d_interpolator = scinew BSplineInterpolator();
    } else{
      ostringstream warn;
      warn << "ERROR:MPM: invalid interpolation type ("<<d_interpolator_type << ")"
           << "Can't be used with axisymmetry at this time \n" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
    }
  } else if(d_interpolator_type=="cpdi"){
    if(d_axisymmetric){
      d_interpolator = scinew axiCpdiInterpolator();
    } else{
      d_interpolator = scinew cpdiInterpolator();
      d_interpolator->setLcrit(d_cpdi_lcrit);
    }
  } else if(d_interpolator_type=="fast_cpdi"){
    if(d_axisymmetric){
      d_interpolator = scinew axiCpdiInterpolator();
    } else{
      d_interpolator = scinew fastCpdiInterpolator();
      d_interpolator->setLcrit(d_cpdi_lcrit);
    }
  } else if(d_interpolator_type=="cpti"){
    if(d_axisymmetric){
      d_interpolator = scinew axiCptiInterpolator();
    } else{
      d_interpolator = scinew cptiInterpolator();
      d_interpolator->setLcrit(d_cpdi_lcrit);
    }
  }
else{
    ostringstream warn;
    warn << "ERROR:MPM: invalid interpolation type ("<<d_interpolator_type << ")"
         << "Valid options are: \n"
         << "linear\n"
         << "gimp\n"
         << "cpdi\n"
         << "cpti\n"
         << "3rdorderBS\n"
         << "4thorderBS\n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
  }
  // Get the size of the vectors associated with the interpolator
  d_8or27=d_interpolator->size();

  mpm_flag_ps->get("extra_solver_flushes", d_extraSolverFlushes);
  mpm_flag_ps->get("boundary_traction_faces", d_bndy_face_txt_list);

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
    dbg << " RefineParticles             = " << d_refineParticles << endl;
    dbg << " XPIC2                       = " << d_XPIC2 << endl;
    dbg << " Use Load Curves             = " << d_useLoadCurves << endl;
    dbg << " Use CBDI boundary condition = " << d_useCBDI << endl;
    dbg << " Use Cohesive Zones          = " << d_useCohesiveZones << endl;
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

  ps->appendElement("interpolator",                       d_interpolator_type);
  ps->appendElement("cpdi_lcrit",                         d_cpdi_lcrit);
  ps->appendElement("AMR",                                d_AMR);
  ps->appendElement("SingleFieldMPM",                     d_SingleFieldMPM);
  ps->appendElement("axisymmetric",                       d_axisymmetric);
  ps->appendElement("withColor",                          d_with_color);
  ps->appendElement("artificial_damping_coeff",           d_artificialDampCoeff);
  ps->appendElement("artificial_viscosity",               d_artificial_viscosity);
  ps->appendElement("artificial_viscosity_heating",       d_artificial_viscosity_heating);
  ps->appendElement("artificial_viscosity_coeff1",        d_artificialViscCoeff1);
  ps->appendElement("artificial_viscosity_coeff2",        d_artificialViscCoeff2);
  ps->appendElement("refine_particles",                   d_refineParticles);
  ps->appendElement("XPIC2",                              d_XPIC2);
  ps->appendElement("use_cohesive_zones",                 d_useCohesiveZones);
  ps->appendElement("use_load_curves",                    d_useLoadCurves);
  ps->appendElement("use_CBDI_boundary_condition",        d_useCBDI);
  ps->appendElement("exactDeformation",                   d_exactDeformation);
  ps->appendElement("DoImplicitHeatConduction",           d_doImplicitHeatConduction);
  ps->appendElement("DoTransientImplicitHeatConduction",  d_doTransientImplicitHeatConduction);
  ps->appendElement("DoExplicitHeatConduction",           d_doExplicitHeatConduction);
  ps->appendElement("DeleteGeometryObjects",              d_deleteGeometryObjects);
  ps->appendElement("DoPressureStabilization",            d_doPressureStabilization);
  ps->appendElement("computeNodalHeatFlux",               d_computeNodalHeatFlux);
  ps->appendElement("computeScaleFactor",                 d_computeScaleFactor);
  ps->appendElement("DoThermalExpansion",                 d_doThermalExpansion);
  ps->appendElement("UseGradientEnhancedVelocityProjection",  d_GEVelProj);
  ps->appendElement("do_grid_reset",                      d_doGridReset);
  ps->appendElement("minimum_particle_mass",              d_min_part_mass);
  ps->appendElement("minimum_subcycles_for_F",            d_min_subcycles_for_F);
  ps->appendElement("minimum_mass_for_acc",               d_min_mass_for_acceleration);
  ps->appendElement("maximum_particle_velocity",          d_max_vel);
  ps->appendElement("UsePrescribedDeformation",           d_prescribeDeformation);

  if(d_prescribeDeformation){
    ps->appendElement("PrescribedDeformationFile",d_prescribedDeformationFile);
  }
//MMS
  ps->appendElement("RunMMSProblem",d_mms_type);
  ps->appendElement("InsertParticles",d_insertParticles);
  if(d_insertParticles){
    ps->appendElement("InsertParticlesFile",d_insertParticlesFile);
  }

  ps->appendElement("do_contact_friction_heating",d_do_contact_friction);
  ps->appendElement("computeNormals",             d_computeNormals);
  ps->appendElement("useLogisticRegression",       d_useLogisticRegression);
  ps->appendElement("computeColinearNormals",     d_computeColinearNormals);
  ps->appendElement("restartOnLargeNodalVelocity",d_restartOnLargeNodalVelocity);
  ps->appendElement("extra_solver_flushes", d_extraSolverFlushes);
  ps->appendElement("boundary_traction_faces", d_bndy_face_txt_list);
  ps->appendElement("do_scalar_diffusion", d_doScalarDiffusion);
  ps->appendElement("d_ndim",                      d_ndim);
}

bool
MPMFlags::doMPMOnLevel(int level, int numLevels) const
{
  return (level >= d_minGridLevel && level <= d_maxGridLevel) ||
          (d_minGridLevel < 0 && level == numLevels + d_minGridLevel);
}

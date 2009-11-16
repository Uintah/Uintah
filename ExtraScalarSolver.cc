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


//----- ExtraScalarSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/ExtraScalarSolver.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PetscSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/RHSSolver.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/ExtraScalarSrc.h>
#include <CCA/Components/Arches/ExtraScalarSrcFactory.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>


using namespace Uintah;
using namespace std;

//****************************************************************************
// Default constructor for ExtraScalarSolver
//****************************************************************************
ExtraScalarSolver::ExtraScalarSolver(const ArchesLabel* label,
                                     const MPMArchesLabel* MAlb,
                                     PhysicalConstants* physConst) :
                                     d_lab(label), d_MAlab(MAlb),
                                     d_physicalConsts(physConst)
{
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
  d_calcExtraScalarSrcs = false;
}

//****************************************************************************
// Destructor
//****************************************************************************
ExtraScalarSolver::~ExtraScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
  VarLabel::destroy(d_scalar_label);
  VarLabel::destroy(d_scalar_temp_label);
  VarLabel::destroy(d_scalar_coef_label);
  VarLabel::destroy(d_scalar_diff_coef_label);
  VarLabel::destroy(d_scalar_nonlin_src_label);
  if (d_calcExtraScalarSrcs){
    for (int i=0; i < static_cast<int>(d_extraScalarSources.size()); i++){
      delete d_extraScalarSources[i];
    }
  }
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ExtraScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params;

  d_discretize = scinew Discretization();
  
  db->getAttribute("label",d_scalar_name);
  
  d_scalar_label = VarLabel::create(d_scalar_name,
                                    CCVariable<double>::getTypeDescription());
  d_scalar_temp_label = VarLabel::create(d_scalar_name+"Temp",
                                    CCVariable<double>::getTypeDescription());
  d_scalar_coef_label = VarLabel::create(d_scalar_name+"Coef",
                                    CCVariable<double>::getTypeDescription());
  d_scalar_diff_coef_label = VarLabel::create(d_scalar_name+"DiffCoef",
                                    CCVariable<double>::getTypeDescription());
  d_scalar_nonlin_src_label = VarLabel::create(d_scalar_name+"NonlinSrc",
                                    CCVariable<double>::getTypeDescription());

  db->require("initial_value",          d_scalar_init_value);
  db->getWithDefault("diffusion",       d_scalar_diffusion,       true);
  db->getWithDefault("density_weighted",d_scalar_density_weighted,true);
  db->getWithDefault("useforDensity",   d_scalar_useforden,       false);
  db->getWithDefault("carbon_balance",  d_carbon_balance,         false);
  db->getWithDefault("sulfur_balance",  d_sulfur_balance,         false);  
  db->getWithDefault("clip_value",      d_clipValue,              100000000000.0);
  db->getWithDefault("noisy_clipping",  d_noisyClipping,          false);

  //I am anticipating that we might have several source terms that 
  // could be read from the table.  For a specific scalar, one would 
  // have to associate the source term variable from the table to this
  // precomputed table source name and add "if" statements accordingly in 
  // explicitsolver.cc and properties.cc (and maybe elsewhere)
  db->getWithDefault("PrecompTabSrcName", d_precompTabSrcName, "null");
  
  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"central-upwind");
  if (conv_scheme == "central-upwind"){
    d_conv_scheme = 0;
  }else{
   throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }
//      else if (conv_scheme == "flux_limited") d_conv_scheme = 1;
   

  if ((d_conv_scheme == 0)&&(!(d_scalar_diffusion))) {
    cout << "WARNING! In the absence of diffusion, convection scheme" << endl;
    cout << "falls back on full upwind for scalar " << d_scalar_name << endl;
  }

  d_calcExtraScalarSrcs = false;
  ProblemSpecP extra_sc_src_db = db->findBlock("sources");
  if (extra_sc_src_db != 0) {
    d_calcExtraScalarSrcs = true;
    for (ProblemSpecP src_db = extra_sc_src_db->findBlock("source");
         src_db != 0; src_db = src_db->findNextBlock("source")) {
      string src_name;
      src_db->getAttribute("label",src_name);
      d_extraScalarSrc = ExtraScalarSrcFactory::create(d_lab, d_MAlab,
                         d_scalar_nonlin_src_label, src_name);
      d_extraScalarSrc->problemSetup(src_db);
      d_extraScalarSources.push_back(d_extraScalarSrc);
    }
  }

/*  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "superbee") d_limiter_type = 0;
      else if (limiter_type == "vanLeer") d_limiter_type = 1;
        else if (limiter_type == "none") {
          d_limiter_type = 2;
          cout << "WARNING! Running central scheme for scalar," << endl;
          cout << "which can be unstable." << endl;
        }
          else if (limiter_type == "central-upwind") d_limiter_type = 3;
            else if (limiter_type == "upwind") d_limiter_type = 4;
              else throw InvalidValue("Flux limiter type "
                                           "not supported: " + limiter_type, __FILE__, __LINE__);
  string boundary_limiter_type;
  d_boundary_limiter_type = 3;
  if (d_limiter_type < 3) {
    db->getWithDefault("boundary_limiter_type",boundary_limiter_type,"central-upwind");
    if (boundary_limiter_type == "none") {
          d_boundary_limiter_type = 2;
          cout << "WARNING! Running central scheme for scalar on the boundaries," << endl;
          cout << "which can be unstable." << endl;
    }
      else if (boundary_limiter_type == "central-upwind") d_boundary_limiter_type = 3;
        else if (boundary_limiter_type == "upwind") d_boundary_limiter_type = 4;
          else throw InvalidValue("Flux limiter type on the boundary"
                                  "not supported: " + boundary_limiter_type, __FILE__, __LINE__);
    d_central_limiter = false;
    if (d_limiter_type < 2)
      db->getWithDefault("central_limiter",d_central_limiter,false);
  }
  }
*/
  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts);
  
//  if (d_doMMS)
//          d_source->problemSetup(db);

  d_rhsSolver = scinew RHSSolver();

  if (d_scalar_diffusion) {
    db->require("turbulentPrandtlNumber",d_turbPrNo);
  }

  d_discretize->setTurbulentPrandtlNumber(d_turbPrNo);
}
//*---------------------------------------------------------------
//* Method for allocating the extra scalars when dummy solve is used
//* for MPMArches.
//* - jeremy t.
//*---------------------------------------------------------------
void ExtraScalarSolver::sched_setInitialGuess(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ExtraScalarSolver::setInitialGuess" + d_scalar_name +timelabels->integrator_step_name;
        
  Task* tsk = scinew Task(taskname, this,
                          &ExtraScalarSolver::setInitialGuess,
                          timelabels);

  tsk->requires(Task::OldDW, d_scalar_label,Ghost::AroundCells, 2);
  tsk->computes(d_scalar_label);

  sched->addTask(tsk, patches, matls);
}
//*---------------------------------------------------------------
//* Implementation of setInitialGuess
//* - jeremy t.
//*---------------------------------------------------------------
void ExtraScalarSolver::setInitialGuess(const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> newscalar;
    constCCVariable<double> oldscalar;

    new_dw->allocateAndPut(newscalar, d_scalar_label, indx, patch);
    newscalar.initialize(0.0);
    old_dw->get(oldscalar, d_scalar_label, indx, patch, Ghost::None, 0);

    newscalar.copyData(oldscalar); //copy old into new. see note.

    //*note that when this was written, the inlet boundary conditions
    // were only applied once at the begining of the solve.  Hence the 
    // need for the actual copy performed here. 
    // -jeremy t. 

  }
}



//****************************************************************************
// Schedule solve of linearized scalar equation
//****************************************************************************
void 
ExtraScalarSolver::solve(SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         const TimeIntegratorLabel* timelabels,
                         bool d_EKTCorrection,
                         bool doing_EKT_now)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : scalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels, d_EKTCorrection,
                          doing_EKT_now);
  
  if (d_calcExtraScalarSrcs){
    for (int i=0; i < static_cast<int>(d_extraScalarSources.size()); i++){
      d_extraScalarSources[i]->sched_addExtraScalarSrc(sched, patches,
                                                       matls, timelabels);
    }
  }

  // Schedule the scalar solve
  // require : scalarIN, scalCoefSBLM, scalNonLinSrcSBLM
  sched_scalarLinearSolve(sched, patches, matls, timelabels, d_EKTCorrection,
                          doing_EKT_now);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ExtraScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels,
                                           bool d_EKTCorrection,
                                           bool doing_EKT_now)
{
  string taskname =  "ExtraScalarSolver::BuildCoeff" + d_scalar_name +
                     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
                          &ExtraScalarSolver::buildLinearMatrix,
                          timelabels, d_EKTCorrection, doing_EKT_now);

  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }

  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn); 
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());

  // This task requires scalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,gac, 1);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->requires(Task::OldDW, d_scalar_label,  gac, 2);
  }else{
    tsk->requires(Task::NewDW, d_scalar_label,  gac, 2);
  }
  
  if (d_scalar_density_weighted){
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gac, 2);
  }

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{
    old_values_dw = Task::NewDW;
  }

  tsk->requires(old_values_dw, d_scalar_label,  gn, 0);
  if (d_scalar_density_weighted){
    tsk->requires(old_values_dw, d_lab->d_densityCPLabel,   gn, 0);
  }

  if (d_scalar_diffusion){
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,  gac, 2);
  }
  
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,   gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,   gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,   gaf, 1);

/*  if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) 
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
    tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
                  d_lab->d_vectorMatl, oams,
                  gac, 1);
  else
    tsk->requires(Task::NewDW, d_lab->d_scalarFluxCompLabel,
                  d_lab->d_vectorMatl, oams,
                  gac, 1);*/


  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    tsk->computes(d_scalar_coef_label, d_lab->d_stencilMatl,      oams);
    tsk->computes(d_scalar_diff_coef_label, d_lab->d_stencilMatl, oams);
    tsk->computes(d_scalar_nonlin_src_label);
//#ifdef divergenceconstraint
//    tsk->computes(d_lab->d_scalDiffCoefSrcLabel);
//#endif
  }
  else {
    tsk->modifies(d_scalar_coef_label,      d_lab->d_stencilMatl,oams);
    tsk->modifies(d_scalar_diff_coef_label, d_lab->d_stencilMatl,oams);
    tsk->modifies(d_scalar_nonlin_src_label);
//#ifdef divergenceconstraint
//    tsk->modifies(d_lab->d_scalDiffCoefSrcLabel);
//#endif
  }

/*  if (doing_EKT_now)
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      tsk->computes(d_lab->d_scalarEKTLabel);
    else
      tsk->modifies(d_lab->d_scalarEKTLabel);*/
    if (timelabels->multiple_steps) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        tsk->computes(d_scalar_temp_label);
      else
        tsk->modifies(d_scalar_temp_label);
    }
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
          tsk->computes(d_scalar_label);

  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ExtraScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw,
                                          const TimeIntegratorLabel* timelabels,
                                          bool d_EKTCorrection,
                                          bool doing_EKT_now)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);

    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
     old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }
    
    //__________________________________
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    // from old_dw get PCELL, DENO, FO
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(constScalarVars.cellType,  d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    
    old_values_dw->get(constScalarVars.old_scalar, d_scalar_label, indx, patch, gn, 0);
    
    CCVariable<double> const_density;
    if (d_scalar_density_weighted){
      old_values_dw->get(constScalarVars.old_density, d_lab->d_densityCPLabel, indx, patch, gn,  0);
      new_dw->get(       constScalarVars.density,     d_lab->d_densityCPLabel, indx, patch, gac, 2);
    }else {
      new_dw->allocateTemporary(const_density,patch, gac,1);
      const_density.initialize(1.0);
      constScalarVars.old_density = const_density;
      constScalarVars.density     = const_density;
    }
  
    CCVariable<double> zero_viscosity;
    if (d_scalar_diffusion){
      new_dw->get(constScalarVars.viscosity, d_lab->d_viscosityCTSLabel, indx, patch,  gac, 2);
    }else {
      new_dw->allocateTemporary(zero_viscosity,patch, gac, 1);
      zero_viscosity.initialize(0.0);
      constScalarVars.viscosity = zero_viscosity;
    }

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->get(constScalarVars.scalar, d_scalar_label, indx, patch,  gac, 2);
    }else{
      new_dw->get(constScalarVars.scalar, d_scalar_label, indx, patch,  gac, 2);
    }
    // for explicit get old values
    new_dw->get(constScalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constScalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constScalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);

 // allocate matrix coeffs
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(scalarVars.scalarCoeff[ii],
                             d_scalar_coef_label, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->allocateAndPut(scalarVars.scalarDiffusionCoeff[ii],
                             d_scalar_diff_coef_label, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(scalarVars.scalarNonlinearSrc,
                           d_scalar_nonlin_src_label, indx, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
/*    new_dw->allocateAndPut(scalarVars.scalarDiffNonlinearSrc,
                           d_lab->d_scalDiffCoefSrcLabel, indx, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);*/
//#endif
  }
  else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(scalarVars.scalarCoeff[ii],
                            d_scalar_coef_label, ii, patch);
      scalarVars.scalarCoeff[ii].initialize(0.0);
      new_dw->getModifiable(scalarVars.scalarDiffusionCoeff[ii],
                            d_scalar_diff_coef_label, ii, patch);
      scalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->getModifiable(scalarVars.scalarNonlinearSrc,
                          d_scalar_nonlin_src_label, indx, patch);
    scalarVars.scalarNonlinearSrc.initialize(0.0);
//#ifdef divergenceconstraint
/*    new_dw->getModifiable(scalarVars.scalarDiffNonlinearSrc,
                          d_lab->d_scalDiffCoefSrcLabel, indx, patch);
    scalarVars.scalarDiffNonlinearSrc.initialize(0.0);*/
//#endif
  }

  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
    new_dw->allocateTemporary(scalarVars.scalarConvectCoeff[ii],  patch);
    scalarVars.scalarConvectCoeff[ii].initialize(0.0);
  }
  new_dw->allocateTemporary(scalarVars.scalarLinearSrc,  patch);
  scalarVars.scalarLinearSrc.initialize(0.0);
  // compute ith component of scalar stencil coefficients
  // inputs : scalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: scalCoefSBLM
  d_discretize->calculateScalarCoeff(patch,
                                     cellinfo, 
                                     &scalarVars, &constScalarVars,
                                     d_conv_scheme);

  // Calculate scalar source terms
  // inputs : [u,v,w]VelocityMS, scalarSP, densityCP, viscosityCTS
  // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
  d_source->calculateExtraScalarSource(pc, patch,
                                  delta_t, cellinfo, 
                                  &scalarVars, &constScalarVars);
  if (d_doMMS)
    d_source->calculateScalarMMSSource(pc, patch,
                                  delta_t, cellinfo, 
                                  &scalarVars, &constScalarVars);
/*    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
                                                  (pc, patch,  cellinfo,
                                                    &scalarVars, &constScalarVars,
                                                  wall_celltypeval, 
                                                  d_limiter_type,
                                                  d_boundary_limiter_type,
                                                  d_central_limiter); 
    } */

    // for scalesimilarity model add scalarflux to the source of scalar eqn.
/*    if (dynamic_cast<const ScaleSimilarityModel*>(d_turbModel)) {
      StencilMatrix<constCCVariable<double> > scalarFlux; //3 point stencil
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
        old_dw->get(scalarFlux[ii], 
                        d_lab->d_scalarFluxCompLabel, ii, patch,
                        gac, 1);
      }
      else
      for (int ii = 0; ii < d_lab->d_vectorMatl->size(); ii++) {
        new_dw->get(scalarFlux[ii], 
                        d_lab->d_scalarFluxCompLabel, ii, patch,
                        gac, 1);
      }
      IntVector indexLow = patch->getFortranCellLowIndex();
      IntVector indexHigh = patch->getFortranCellHighIndex();
      
      // set density for the whole domain
      
      
      // Store current cell
      double sue, suw, sun, sus, sut, sub;
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector prevXCell(colX-1, colY, colZ);
            IntVector prevYCell(colX, colY-1, colZ);
            IntVector prevZCell(colX, colY, colZ-1);
            IntVector nextXCell(colX+1, colY, colZ);
            IntVector nextYCell(colX, colY+1, colZ);
            IntVector nextZCell(colX, colY, colZ+1);
            
            sue = 0.5*cellinfo->sns[colY]*cellinfo->stb[colZ]*
              ((scalarFlux[0])[currCell]+(scalarFlux[0])[nextXCell]);
            suw = 0.5*cellinfo->sns[colY]*cellinfo->stb[colZ]*
              ((scalarFlux[0])[prevXCell]+(scalarFlux[0])[currCell]);
            sun = 0.5*cellinfo->sew[colX]*cellinfo->stb[colZ]*
              ((scalarFlux[1])[currCell]+ (scalarFlux[1])[nextYCell]);
            sus = 0.5*cellinfo->sew[colX]*cellinfo->stb[colZ]*
              ((scalarFlux[1])[currCell]+(scalarFlux[1])[prevYCell])diff_;
            sut = 0.5*cellinfo->sns[colY]*cellinfo->sew[colX]*
              ((scalarFlux[2])[currCell]+ (scalarFlux[2])[nextZCell]);
            sub = 0.5*cellinfo->sns[colY]*cellinfo->sew[colX]*
              ((scalarFlux[2])[currCell]+ (scalarFlux[2])[prevZCell]);
#if 1
            scalarVars.scalarNonlinearSrc[currCell] += suw-sue+sus-sun+sub-sut;
#ifdef divergenceconstraint
            scalarVars.scalarDiffNonlinearSrc[currCell] = suw-sue+sus-sun+sub-sut;
#endif
#endif
          }
        }
      }
    }*/
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, scalCoefSBLM
    // outputs: scalCoefSBLM
    
    
    if (d_boundaryCondition->anyArchesPhysicalBC()) {
      d_boundaryCondition->scalarBC(patch,
                                    &scalarVars, &constScalarVars);
      /*if (d_boundaryCondition->getIntrusionBC())
        d_boundaryCondition->intrusionScalarBC(pc, patch, cellinfo,
                                               &scalarVars, &constScalarVars);*/
    }
    // apply multimaterial intrusion wallbc
    if (d_MAlab)
      d_boundaryCondition->mmscalarWallBC(patch, cellinfo,
                                          &scalarVars, &constScalarVars);
    
    d_source->modifyScalarMassSource(pc, patch, delta_t,
                                     &scalarVars, &constScalarVars,
                                     d_conv_scheme);
    
    // Calculate the scalar diagonal terms
    // inputs : scalCoefSBLM, scalLinSrcSBLM
    // outputs: scalCoefSBLM
    d_discretize->calculateScalarDiagonal(patch, &scalarVars);

    /*CCVariable<double> scalar;
    if (doing_EKT_now) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        new_dw->allocateAndPut(scalar, d_lab->d_scalarEKTLabel, indx, patch);
      else
        new_dw->getModifiable(scalar, d_lab->d_scalarEKTLabel,  indx, patch);

        new_dw->copyOut(scalar, d_lab->d_scalarSPLabel, indx, patch);
    }*/
    CCVariable<double> scalar_temp;
    if (timelabels->multiple_steps) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
        new_dw->allocateAndPut(scalar_temp, d_scalar_temp_label, indx, patch);
        old_dw->copyOut(scalar_temp,        d_scalar_label,      indx, patch);
      }else {
        new_dw->getModifiable(scalar_temp,  d_scalar_temp_label, indx, patch);
        new_dw->copyOut(scalar_temp,        d_scalar_label,      indx, patch);
      }
    }
    CCVariable<double> new_scalar;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(new_scalar, d_scalar_label, indx, patch);
      old_dw->copyOut(new_scalar,        d_scalar_label, indx, patch);
    }
  }
}


//****************************************************************************
// Schedule linear solve of scalar
//****************************************************************************
void
ExtraScalarSolver::sched_scalarLinearSolve(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels,
                                           bool d_EKTCorrection,
                                           bool doing_EKT_now)
{
  string taskname =  "ExtraScalarSolver::ScalarLinearSolve" + d_scalar_name +
                     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
                          &ExtraScalarSolver::scalarLinearSolve,
                          timelabels, d_EKTCorrection, doing_EKT_now);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){ 
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn); 
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel,     gac, 1);

  if (d_scalar_density_weighted){
    tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, gn, 0);
  }
  
  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_scalar_temp_label, gac, 1);
  }else{
    tsk->requires(Task::OldDW, d_scalar_label,      gac, 1);
  }
  tsk->requires(Task::NewDW, d_scalar_coef_label, 
                d_lab->d_stencilMatl, oams, gn, 0);
  tsk->requires(Task::NewDW, d_scalar_nonlin_src_label, gn, 0);


  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel,gn, 0);
  }    

/*  if (doing_EKT_now)
    tsk->modifies(d_lab->d_scalarEKTLabel);
  else */
    tsk->modifies(d_scalar_label);
/*  if (timelabels->recursion)
    tsk->computes(d_lab->d_ScalarClippedLabel);*/
  
  
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual scalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ExtraScalarSolver::scalarLinearSolve(const ProcessorGroup* pc,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     const TimeIntegratorLabel* timelabels,
                                     bool d_EKTCorrection,
                                     bool doing_EKT_now)
{
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }
  
  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_sharedState->get_delt_label() );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->
                    getArchesMaterial(archIndex)->getDWIndex(); 
    ArchesVariables scalarVars;
    ArchesConstVariables constScalarVars;

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    CCVariable<double> const_density;
    if (d_scalar_density_weighted){
      new_dw->get(constScalarVars.density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);
    }else {
      const_density.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
      const_density.initialize(1.0);
      constScalarVars.density_guess = const_density;
    }

    if (timelabels->multiple_steps){
      new_dw->get(constScalarVars.old_scalar, d_scalar_temp_label, indx, patch, gac, 1);
    }else{
      old_dw->get(constScalarVars.old_scalar, d_scalar_label,      indx, patch, gac, 1);
    }
    // for explicit calculation
/*    if (doing_EKT_now)
      new_dw->getModifiable(scalarVars.scalar, d_lab->d_scalarEKTLabel, indx, patch);
    else*/
      new_dw->getModifiable(scalarVars.scalar, d_scalar_label, indx, patch);

    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++){
      new_dw->get(constScalarVars.scalarCoeff[ii], d_scalar_coef_label, ii, patch, gn, 0);
    }
    
    new_dw->get(constScalarVars.scalarNonlinearSrc,d_scalar_nonlin_src_label, indx, patch, gn, 0);
    new_dw->get(constScalarVars.cellType,         d_lab->d_cellTypeLabel,     indx,patch, gac, 1);
    if (d_MAlab) {
      new_dw->get(constScalarVars.voidFraction, d_lab->d_mmgasVolFracLabel,indx, patch, gn, 0);
    }

    // make it a separate task later

    if (d_MAlab){
      d_boundaryCondition->scalarLisolve_mm(patch, delta_t, 
                                            &scalarVars, &constScalarVars,
                                            cellinfo);
    }else{
      d_rhsSolver->scalarLisolve(pc, patch, delta_t, 
                                    &scalarVars, &constScalarVars,
                                    cellinfo);
    }
    
  double scalar_clipped = 0.0;
  double epsilon = 1.0e-15;
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
        IntVector currCell(ii,jj,kk);
        if (scalarVars.scalar[currCell] > d_clipValue) {
          if (scalarVars.scalar[currCell] > d_clipValue + epsilon) {
            scalar_clipped = 1.0;
            if (d_noisyClipping){
              cout << "Clipping extra scalar value! " << currCell
                   << " , scalar value was " << scalarVars.scalar[currCell]  << endl;
            }
          }
          scalarVars.scalar[currCell] = d_clipValue;
        }  
        else if (scalarVars.scalar[currCell] < 0.0) {
          if (scalarVars.scalar[currCell] < - epsilon) {
            scalar_clipped = 1.0;
            if (d_noisyClipping){
              cout << "scalar got clipped to 0 at " << currCell
                   << " , scalar value was " << scalarVars.scalar[currCell] << endl;
            } 
          }
          scalarVars.scalar[currCell] = 0.0;
        }
      }  //z
    }  //y
  }  // x

// Outlet bc is done here not to change old scalar
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC()))
    d_boundaryCondition->scalarOutletPressureBC(patch,
                                        &scalarVars, &constScalarVars);

  }
}


/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


//----- ReactiveScalarSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/ReactiveScalarSolver.h>
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
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
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
// Default constructor for ReactiveScalarSolver
//****************************************************************************
ReactiveScalarSolver::ReactiveScalarSolver(const ArchesLabel* label,
                           const MPMArchesLabel* MAlb,
                           TurbulenceModel* turb_model,
                           BoundaryCondition* bndry_cond,
                           PhysicalConstants* physConst) :
                                 d_lab(label), d_MAlab(MAlb),
                                 d_turbModel(turb_model), 
                                 d_boundaryCondition(bndry_cond),
                                 d_physicalConsts(physConst)
{
  d_discretize = 0;
  d_source = 0;
  d_rhsSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
ReactiveScalarSolver::~ReactiveScalarSolver()
{
  delete d_discretize;
  delete d_source;
  delete d_rhsSolver;
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
ReactiveScalarSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("ReactiveScalarSolver");

  d_discretize = scinew Discretization();

  string conv_scheme;
  db->getWithDefault("convection_scheme",conv_scheme,"central-upwind");
  if (conv_scheme == "central-upwind"){
    d_conv_scheme = 0;
  }else if (conv_scheme == "flux_limited"){
   d_conv_scheme = 1;
  }else{
   throw InvalidValue("Convection scheme not supported: " + conv_scheme, __FILE__, __LINE__);
  }
  
  string limiter_type;
  if (d_conv_scheme == 1) {
    db->getWithDefault("limiter_type",limiter_type,"superbee");
    if (limiter_type == "superbee"){
      d_limiter_type = 0;
    }else if (limiter_type == "vanLeer"){
      d_limiter_type = 1;
    }else if (limiter_type == "none") {
      d_limiter_type = 2;
      cout << "WARNING! Running central scheme for scalar," << endl;
      cout << "which can be unstable." << endl;
    }else if (limiter_type == "central-upwind"){
      d_limiter_type = 3;
    }else if (limiter_type == "upwind"){
      d_limiter_type = 4;
    }else{
      throw InvalidValue("Flux limiter type not supported: " + limiter_type, __FILE__, __LINE__);
    }
    string boundary_limiter_type;
    d_boundary_limiter_type = 3;
    if (d_limiter_type < 3) {
      db->getWithDefault("boundary_limiter_type",boundary_limiter_type,"central-upwind");
      if (boundary_limiter_type == "none") {
        d_boundary_limiter_type = 2;
        cout << "WARNING! Running central scheme for scalar on the boundaries," << endl;
        cout << "which can be unstable." << endl;
      }else if (boundary_limiter_type == "central-upwind"){
        d_boundary_limiter_type = 3;
      }else if (boundary_limiter_type == "upwind"){
        d_boundary_limiter_type = 4;
      }else{
        throw InvalidValue("Flux limiter type on the boundary"
                                    "not supported: " + boundary_limiter_type, __FILE__, __LINE__);
      }
      d_central_limiter = false;
      if (d_limiter_type < 2){
        db->getWithDefault("central_limiter",d_central_limiter,false);
      }
    }
  }

  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts);
  
  if (d_doMMS){
    d_source->problemSetup(db);
  }
  d_rhsSolver = scinew RHSSolver();

  d_dynScalarModel = d_turbModel->getDynScalarModel();
  double model_turbPrNo;
  model_turbPrNo = d_turbModel->getTurbulentPrandtlNumber();

  // see if Prandtl number gets overridden here
  d_turbPrNo = 0.0;
  if (!(d_dynScalarModel)) {
    if (db->findBlock("turbulentPrandtlNumber")){
      db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
    }
    
    // if it is not set in both places
    if ((d_turbPrNo == 0.0)&&(model_turbPrNo == 0.0)){
      throw InvalidValue("Turbulent Prandtl number is not specified for"
                             "mixture fraction ", __FILE__, __LINE__);
    }
    // if it is set in turbulence model
    else if (d_turbPrNo == 0.0){
      d_turbPrNo = model_turbPrNo;
    }
  }

  d_discretize->setTurbulentPrandtlNumber(d_turbPrNo);
}

//****************************************************************************
// Schedule solve of linearized reactscalar equation
//****************************************************************************
void 
ReactiveScalarSolver::solve(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const TimeIntegratorLabel* timelabels,
                            bool d_EKTCorrection,
                            bool doing_EKT_now)
{
  //computes stencil coefficients and source terms
  // requires : scalarIN, [u,v,w]VelocitySPBC, densityIN, viscosityIN
  // computes : reactscalCoefSBLM, scalLinSrcSBLM, scalNonLinSrcSBLM
  sched_buildLinearMatrix(sched, patches, matls, timelabels, d_EKTCorrection,
                          doing_EKT_now);
  
  // Schedule the scalar solve
  // require : scalarIN, reactscalCoefSBLM, scalNonLinSrcSBLM
  sched_reactscalarLinearSolve(sched, patches, matls, timelabels, 
                               d_EKTCorrection, doing_EKT_now);
}

//****************************************************************************
// Schedule build of linear matrix
//****************************************************************************
void 
ReactiveScalarSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels,
                                              bool d_EKTCorrection,
                                              bool doing_EKT_now)
{
  string taskname =  "ReactiveScalarSolver::BuildCoeff" +
                     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
                          &ReactiveScalarSolver::buildLinearMatrix,
                          timelabels, d_EKTCorrection, doing_EKT_now);


  Task::WhichDW parent_old_dw;
  if (timelabels->recursion) parent_old_dw = Task::ParentOldDW;
  else parent_old_dw = Task::OldDW;

  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  
  // This task requires reactscalar and density from old time step for transient
  // calculation
  //DataWarehouseP old_dw = new_dw->getTop();
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
    
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_reactscalarSPLabel,gac, 2);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,    gac, 2);

  Task::WhichDW old_values_dw;
  if (timelabels->use_old_values){
    old_values_dw = parent_old_dw;
  }else{ 
    old_values_dw = Task::NewDW;
  }
  tsk->requires(old_values_dw, d_lab->d_reactscalarSPLabel, gn, 0);
  tsk->requires(old_values_dw, d_lab->d_densityCPLabel,     gn, 0);

  if (d_dynScalarModel){
    tsk->requires(Task::NewDW, d_lab->d_reactScalarDiffusivityLabel,gac, 2);
  }else{
    tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,          gac, 2);
  }

  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);

  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))){
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSRCINLabel,  gn, 0);
  }else{
    tsk->requires(Task::NewDW, d_lab->d_reactscalarSRCINLabel,  gn, 0);
  }
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    tsk->computes(d_lab->d_reactscalCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_reactscalDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->computes(d_lab->d_reactscalNonLinSrcSBLMLabel);
  }else {
    tsk->modifies(d_lab->d_reactscalCoefSBLMLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_reactscalDiffCoefLabel, d_lab->d_stencilMatl, oams);
    tsk->modifies(d_lab->d_reactscalNonLinSrcSBLMLabel);
  }
  if (doing_EKT_now){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->computes(d_lab->d_reactscalarEKTLabel);
    }else{
      tsk->modifies(d_lab->d_reactscalarEKTLabel);
    }
  }
  sched->addTask(tsk, patches, matls);
}

      
//****************************************************************************
// Actually build linear matrix
//****************************************************************************
void ReactiveScalarSolver::buildLinearMatrix(const ProcessorGroup* pc,
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
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    

    // from old_dw get PCELL, DENO, FO
    DataWarehouse* old_values_dw;
    if (timelabels->use_old_values){
      old_values_dw = parent_old_dw;
    }else{
      old_values_dw = new_dw;
    }
    
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None;
        
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    new_dw->get(constReactscalarVars.cellType,           d_lab->d_cellTypeLabel,      indx, patch, gac, 1);
    old_values_dw->get(constReactscalarVars.old_scalar,  d_lab->d_reactscalarSPLabel, indx, patch, gn, 0);
    old_values_dw->get(constReactscalarVars.old_density, d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    
    new_dw->get(constReactscalarVars.density, d_lab->d_densityCPLabel,    indx, patch,  gac, 2);
    new_dw->get(constReactscalarVars.scalar, d_lab->d_reactscalarSPLabel, indx, patch,  gac, 2);
    
    if (d_dynScalarModel){
      new_dw->get(constReactscalarVars.viscosity,
                  d_lab->d_reactScalarDiffusivityLabel, indx, patch, gac, 2);
    }else{
      new_dw->get(constReactscalarVars.viscosity, 
                  d_lab->d_viscosityCTSLabel,           indx, patch, gac, 2);
    }
    
    // for explicit get old values
    new_dw->get(constReactscalarVars.uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constReactscalarVars.vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(constReactscalarVars.wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);

    // computes reaction scalar source term in properties
    if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))){
      old_dw->get(constReactscalarVars.reactscalarSRC,
                  d_lab->d_reactscalarSRCINLabel, indx, patch, gn, 0);
    }else{
      new_dw->get(constReactscalarVars.reactscalarSRC,
                  d_lab->d_reactscalarSRCINLabel, indx, patch, gn, 0);
    }


  // allocate matrix coeffs
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&((!(d_EKTCorrection))||((d_EKTCorrection)&&(doing_EKT_now)))) {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->allocateAndPut(reactscalarVars.scalarCoeff[ii],
                             d_lab->d_reactscalCoefSBLMLabel, ii, patch);
      reactscalarVars.scalarCoeff[ii].initialize(0.0);
      
      
      new_dw->allocateAndPut(reactscalarVars.scalarDiffusionCoeff[ii],
                             d_lab->d_reactscalDiffCoefLabel, ii, patch);
      reactscalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->allocateAndPut(reactscalarVars.scalarNonlinearSrc,
                           d_lab->d_reactscalNonLinSrcSBLMLabel,indx, patch);
    reactscalarVars.scalarNonlinearSrc.initialize(0.0);
  }
  else {
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
      new_dw->getModifiable(reactscalarVars.scalarCoeff[ii],
                            d_lab->d_reactscalCoefSBLMLabel, ii, patch);
      reactscalarVars.scalarCoeff[ii].initialize(0.0);
      
      new_dw->getModifiable(reactscalarVars.scalarDiffusionCoeff[ii],
                            d_lab->d_reactscalDiffCoefLabel, ii, patch);
      reactscalarVars.scalarDiffusionCoeff[ii].initialize(0.0);
    }
    new_dw->getModifiable(reactscalarVars.scalarNonlinearSrc,
                          d_lab->d_reactscalNonLinSrcSBLMLabel,indx, patch);
    reactscalarVars.scalarNonlinearSrc.initialize(0.0);
  }

  for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++) {
    new_dw->allocateTemporary(reactscalarVars.scalarConvectCoeff[ii],  patch);
    reactscalarVars.scalarConvectCoeff[ii].initialize(0.0);
  }
  new_dw->allocateTemporary(reactscalarVars.scalarLinearSrc,  patch);
  reactscalarVars.scalarLinearSrc.initialize(0.0);
 
  // compute ith component of reactscalar stencil coefficients
  // inputs : reactscalarSP, [u,v,w]VelocityMS, densityCP, viscosityCTS
  // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarCoeff(patch,
                                       cellinfo, 
                                       &reactscalarVars, &constReactscalarVars,
                                       d_conv_scheme);

    // Calculate reactscalar source terms
    // inputs : [u,v,w]VelocityMS, reactscalarSP, densityCP, viscosityCTS
    // outputs: scalLinSrcSBLM, scalNonLinSrcSBLM
    d_source->calculateScalarSource(pc, patch,
                                    delta_t, cellinfo, 
                                    &reactscalarVars, &constReactscalarVars);
    d_source->addReactiveScalarSource(pc, patch,
                                    delta_t, cellinfo, 
                                    &reactscalarVars, &constReactscalarVars);
    if (d_conv_scheme > 0) {
      int wall_celltypeval = d_boundaryCondition->wallCellType();
      d_discretize->calculateScalarFluxLimitedConvection
                                                  (patch,  cellinfo,
                                                    &reactscalarVars,
                                                  &constReactscalarVars,
                                                  wall_celltypeval, 
                                                  d_limiter_type, 
                                                  d_boundary_limiter_type,
                                                  d_central_limiter); 
    }
    // Calculate the scalar boundary conditions
    // inputs : scalarSP, reactscalCoefSBLM
    // outputs: reactscalCoefSBLM
    if (d_boundaryCondition->anyArchesPhysicalBC()){
      d_boundaryCondition->scalarBC(patch, 
                                  &reactscalarVars, &constReactscalarVars);
    }
  // apply multimaterial intrusion wallbc
    if (d_MAlab){
      d_boundaryCondition->mmscalarWallBC(patch, cellinfo,
                                &reactscalarVars, &constReactscalarVars);
    }    

    // similar to mascal
    // inputs :
    // outputs:
    d_source->modifyScalarMassSource(pc, patch, delta_t,
                                     &reactscalarVars, &constReactscalarVars,
                                     d_conv_scheme);
    
    // Calculate the reactscalar diagonal terms
    // inputs : reactscalCoefSBLM, scalLinSrcSBLM
    // outputs: reactscalCoefSBLM
    d_discretize->calculateScalarDiagonal(patch, &reactscalarVars);

    CCVariable<double> reactscalar;
    if (doing_EKT_now) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        new_dw->allocateAndPut(reactscalar, d_lab->d_reactscalarEKTLabel, indx, patch);
      }else{
        new_dw->getModifiable(reactscalar, d_lab->d_reactscalarEKTLabel,  indx, patch);
      }
      new_dw->copyOut(reactscalar, d_lab->d_reactscalarSPLabel, indx, patch);
    }
  }
}


//****************************************************************************
// Schedule linear solve of reactscalar
//****************************************************************************
void
ReactiveScalarSolver::sched_reactscalarLinearSolve(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls,
                                                   const TimeIntegratorLabel* timelabels,
                                                   bool d_EKTCorrection,
                                                   bool doing_EKT_now)
{
  string taskname =  "ReactiveScalarSolver::ScalarLinearSolve" + 
                     timelabels->integrator_step_name;
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
                          &ReactiveScalarSolver::reactscalarLinearSolve,
                          timelabels, d_EKTCorrection, doing_EKT_now);
  
  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{ 
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  Task::DomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,     gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityGuessLabel, gn,  0);

  if (timelabels->multiple_steps){
    tsk->requires(Task::NewDW, d_lab->d_reactscalarTempLabel, gac, 1);
  }else{
    tsk->requires(Task::OldDW, d_lab->d_reactscalarSPLabel,   gac, 1);
  }
  
  tsk->requires(Task::NewDW, d_lab->d_reactscalCoefSBLMLabel, 
                             d_lab->d_stencilMatl, oams, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_reactscalNonLinSrcSBLMLabel, gn, 0);

  if (doing_EKT_now){
    tsk->modifies(d_lab->d_reactscalarEKTLabel);
  }else{ 
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  }
  if (timelabels->recursion){
    tsk->computes(d_lab->d_ReactScalarClippedLabel);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Actual reactscalar solve .. may be changed after recursive tasks are added
//****************************************************************************
void 
ReactiveScalarSolver::reactscalarLinearSolve(const ProcessorGroup* pc,
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
    ArchesVariables reactscalarVars;
    ArchesConstVariables constReactscalarVars;

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
  
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constReactscalarVars.density_guess, d_lab->d_densityGuessLabel, indx, patch, gn, 0);

    if (timelabels->multiple_steps){
      new_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarTempLabel, indx, patch, gac, 1);
    }else{
      old_dw->get(constReactscalarVars.old_scalar, d_lab->d_reactscalarSPLabel,   indx, patch, gac, 1);
    }
    
    // for explicit calculation
    if (doing_EKT_now){
      new_dw->getModifiable(reactscalarVars.scalar, d_lab->d_reactscalarEKTLabel, indx, patch);
    }else{
      new_dw->getModifiable(reactscalarVars.scalar, d_lab->d_reactscalarSPLabel,  indx, patch);
    }
    
    for (int ii = 0; ii < d_lab->d_stencilMatl->size(); ii++){
      new_dw->get(constReactscalarVars.scalarCoeff[ii],
                  d_lab->d_reactscalCoefSBLMLabel, ii, patch, gn, 0);
    }
    
    new_dw->get(constReactscalarVars.scalarNonlinearSrc,
                d_lab->d_reactscalNonLinSrcSBLMLabel, indx, patch, gn, 0);

    new_dw->get(constReactscalarVars.cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

    // make it a separate task later
    d_rhsSolver->scalarLisolve(pc, patch, delta_t, 
                               &reactscalarVars, &constReactscalarVars,
                               cellinfo);

    double reactscalar_clipped = 0.0;
    double epsilon = 1.0e-15;
        
        
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;

      if (reactscalarVars.scalar[c] > 1.0) {
        if (reactscalarVars.scalar[c] > 1.0 + epsilon) {
          reactscalar_clipped = 1.0;
          cout << "reactscalar got clipped to 1 at " << c << " , reactscalar value was " << reactscalarVars.scalar[c] << " , density guess was " << constReactscalarVars.density_guess[c] << endl;
        }
        reactscalarVars.scalar[c] = 1.0;
      }  
      else if (reactscalarVars.scalar[c] < 0.0) {
        if (reactscalarVars.scalar[c] < - epsilon) {
          reactscalar_clipped = 1.0;
          cout << "reactscalar got clipped to 0 at " << c << " , reactscalar value was " << reactscalarVars.scalar[c] << " , density guess was " << constReactscalarVars.density_guess[c] << endl;
          cout << "Try setting <scalarUnderflowCheck>true</scalarUnderflowCheck> in the <ARCHES> section of the input file, but it would only help for first time substep if RKSSP is used" << endl;
        }
        reactscalarVars.scalar[c] = 0.0;
      }
    }

    if (timelabels->recursion){
      new_dw->put(max_vartype(reactscalar_clipped), d_lab->d_ReactScalarClippedLabel);
    }

// Outlet bc is done here not to change old scalar
    if ((d_boundaryCondition->getOutletBC())||
        (d_boundaryCondition->getPressureBC())){
      d_boundaryCondition->scalarOutletPressureBC(patch, &reactscalarVars, &constReactscalarVars);
    }
  }
}


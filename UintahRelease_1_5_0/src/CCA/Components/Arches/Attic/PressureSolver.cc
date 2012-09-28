/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


//----- PressureSolver.cc ----------------------------------------------

#include <sci_defs/hypre_defs.h>

#include <CCA/Components/Arches/PressureSolver.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PetscSolver.h>
#ifdef HAVE_HYPRE
#include <CCA/Components/Arches/HypreSolver.h>
#endif
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace std;


// ****************************************************************************
// Default constructor for PressureSolver
// ****************************************************************************
PressureSolver::PressureSolver(ArchesLabel* label,
                               const MPMArchesLabel* MAlb,
                               BoundaryCondition* bndry_cond,
                               PhysicalConstants* phys_const,
                               const ProcessorGroup* myworld):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_boundaryCondition(bndry_cond),
                                     d_physicalConsts(phys_const),
                                     d_myworld(myworld)
{
  d_perproc_patches=0;
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0; 
  d_construct_solver_obj = true;
  d_iteration = 0;
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PressureSolver::~PressureSolver()
{
  if ( !d_always_construct_A ) { 
    // destroy A, x, and B
    d_linearSolver->destroyMatrix();
  }

  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

// ****************************************************************************
// Problem Setup
// ****************************************************************************
void 
PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PressureSolver");
  d_pressRef = d_physicalConsts->getRefPoint();
  db->getWithDefault("normalize_pressure",      d_norm_pres, false);
  db->getWithDefault("do_only_last_projection", d_do_only_last_projection, false);

  db->getWithDefault( "always_construct_A", d_always_construct_A, true ); 
  d_construct_A = true;  // Must always be true @ start.  

  d_discretize = scinew Discretization();

  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts);
  if (d_doMMS){
    d_source->problemSetup(db);
  }
  string linear_sol;
  db->require("linear_solver", linear_sol);
  if (linear_sol == "petsc"){
    d_linearSolver = scinew PetscSolver(d_myworld);
  }
#ifdef HAVE_HYPRE
  else if (linear_sol == "hypre"){
    d_linearSolver = scinew HypreSolver(d_myworld);
  }
#endif
  else {
    throw InvalidValue("Linear solver option"
                       " not supported" + linear_sol, __FILE__, __LINE__);
  }
  d_linearSolver->problemSetup(db);
}
// ****************************************************************************
// Schedule solve of linearized pressure equation
// ****************************************************************************
void PressureSolver::sched_solve(const LevelP& level,
                           SchedulerP& sched,
                           const TimeIntegratorLabel* timelabels,
                           bool extraProjection,
                           bool d_EKTCorrection,
                           bool doing_EKT_now)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  sched_buildLinearMatrix(sched, patches, matls, timelabels, extraProjection,
                         d_EKTCorrection, doing_EKT_now);

  sched_pressureLinearSolve(level, sched, timelabels, extraProjection,
                            d_EKTCorrection, doing_EKT_now);

  if ((d_MAlab)&&(!(extraProjection))) {
      sched_addHydrostaticTermtoPressure(sched, patches, matls, timelabels);
  }
}

// ****************************************************************************
// Schedule build of linear matrix
// ****************************************************************************
void 
PressureSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool extraProjection,
                                        bool d_EKTCorrection,
                                        bool doing_EKT_now)
{
  //  build pressure equation coefficients and source
  string taskname =  "PressureSolver::buildLinearMatrix" +
                     timelabels->integrator_step_name;
  if (extraProjection){
    taskname += "extraProjection";
  }
  if (doing_EKT_now){
    taskname += "EKTnow";
  }
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::buildLinearMatrix,
                          timelabels, extraProjection,
                          d_EKTCorrection, doing_EKT_now);
    

  Task::WhichDW parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = Task::ParentOldDW;
  }else{
    parent_old_dw = Task::OldDW;
  }
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  
  tsk->requires(parent_old_dw, d_lab->d_sharedState->get_delt_label());
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->requires(Task::OldDW, timelabels->pressure_guess, gn, 0);
  }else{
    tsk->requires(Task::NewDW, timelabels->pressure_guess, gn, 0);
  }
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_uVelRhoHatLabel,     gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelRhoHatLabel,     gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelRhoHatLabel,     gaf, 1);
  // get drhodt that goes in the rhs of the pressure equation
  tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,   gn, 0);
#ifdef divergenceconstraint
  tsk->requires(Task::NewDW, d_lab->d_divConstraintLabel,  gn, 0);
#endif
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gac, 1);
  }

  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
      &&(((!(extraProjection))&&(!(d_EKTCorrection)))
         ||((d_EKTCorrection)&&(doing_EKT_now)))) {
    tsk->computes(d_lab->d_presCoefPBLMLabel);
    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);
  } else {
    tsk->modifies(d_lab->d_presCoefPBLMLabel);
    tsk->modifies(d_lab->d_presNonLinSrcPBLMLabel);
  }
  

  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Actually build of linear matrix for pressure equation
// ****************************************************************************
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /* matls */,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection,
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
    ArchesVariables pressureVars;
    ArchesConstVariables constPressureVars;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->get(constPressureVars.pressure, timelabels->pressure_guess, indx, patch, gn, 0);
    }else{
      new_dw->get(constPressureVars.pressure, timelabels->pressure_guess, indx, patch, gn, 0);
    }
    new_dw->get(constPressureVars.cellType,     d_lab->d_cellTypeLabel,    indx, patch, gac, 1);
    new_dw->get(constPressureVars.density,      d_lab->d_densityCPLabel,   indx, patch, gac, 1);
    new_dw->get(constPressureVars.uVelRhoHat,   d_lab->d_uVelRhoHatLabel,  indx, patch, gaf, 1);
    new_dw->get(constPressureVars.vVelRhoHat,   d_lab->d_vVelRhoHatLabel,  indx, patch, gaf, 1);
    new_dw->get(constPressureVars.wVelRhoHat,   d_lab->d_wVelRhoHatLabel,  indx, patch, gaf, 1);
    new_dw->get(constPressureVars.filterdrhodt, d_lab->d_filterdrhodtLabel,indx, patch, gn, 0);
#ifdef divergenceconstraint
    new_dw->get(constPressureVars.divergence,   d_lab->d_divConstraintLabel,indx, patch, gn, 0);
#endif

    if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        &&(((!(extraProjection))&&(!(d_EKTCorrection)))
           ||((d_EKTCorrection)&&(doing_EKT_now)))){
      new_dw->allocateAndPut(pressureVars.pressCoeff,        d_lab->d_presCoefPBLMLabel,      indx, patch);
      new_dw->allocateAndPut(pressureVars.pressNonlinearSrc, d_lab->d_presNonLinSrcPBLMLabel, indx, patch);
    }else{
      new_dw->getModifiable(pressureVars.pressCoeff,         d_lab->d_presCoefPBLMLabel,      indx, patch);
      new_dw->getModifiable(pressureVars.pressNonlinearSrc,  d_lab->d_presNonLinSrcPBLMLabel, indx, patch);
    }
    
    new_dw->allocateTemporary(pressureVars.pressLinearSrc,  patch);
    pressureVars.pressLinearSrc.initialize(0.0);   
    pressureVars.pressNonlinearSrc.initialize(0.0);   

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

  

    //__________________________________

    calculatePressureCoeff(patch, cellinfo, &pressureVars, &constPressureVars);


    // Modify pressure coefficients for multimaterial formulation
    if (d_MAlab) {
      new_dw->get(constPressureVars.voidFraction,
                  d_lab->d_mmgasVolFracLabel, indx, patch,
                  gac, 1);

      mmModifyPressureCoeffs(patch, &pressureVars,  &constPressureVars);

    }

 
    d_source->calculatePressureSourcePred(pc, patch, delta_t,
                                          cellinfo, &pressureVars,
                                          &constPressureVars,
                                          doing_EKT_now);

    if (d_doMMS){
      d_source->calculatePressMMSSourcePred(pc, patch, delta_t,
                                            cellinfo, &pressureVars,
                                            &constPressureVars);
    }

    // do multimaterial bc; this is done before 
    // calculatePressDiagonal because unlike the outlet
    // boundaries in the explicit projection, we want to 
    // show the effect of AE, etc. in AP for the 
    // intrusion boundaries    
    if (d_MAlab){
      d_boundaryCondition->mmpressureBC(new_dw, patch,
                                        &pressureVars, &constPressureVars);
    }
    // Calculate Pressure Diagonal
    d_discretize->calculatePressDiagonal(patch,&pressureVars);

    d_boundaryCondition->pressureBC(patch, indx, &pressureVars,&constPressureVars);
  }

}

// ****************************************************************************
// Schedule solver for linear matrix
// ****************************************************************************
void 
PressureSolver::sched_pressureLinearSolve(const LevelP& level,
                                          SchedulerP& sched,
                                          const TimeIntegratorLabel* timelabels,
                                          bool extraProjection,
                                          bool d_EKTCorrection,
                                          bool doing_EKT_now)
{
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;

  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();

  string taskname =  "PressureSolver::PressLinearSolve_all" + 
                     timelabels->integrator_step_name;
  if (extraProjection) taskname += "extraProjection";
  if (doing_EKT_now) taskname += "EKTnow";
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::pressureLinearSolve_all,
                          timelabels, extraProjection,
                          d_EKTCorrection, doing_EKT_now);

  // Requires
  // coefficient for the variable for which solve is invoked
  Ghost::GhostType  gn = Ghost::None;
  if (!((d_pressure_correction)||(extraProjection)
        ||((d_EKTCorrection)&&(doing_EKT_now)))){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, timelabels->pressure_guess, gn, 0);
    }else{
      tsk->requires(Task::NewDW, timelabels->pressure_guess, gn, 0);
    }
  }
  tsk->requires(Task::NewDW, d_lab->d_presCoefPBLMLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,gn, 0);


  if ((extraProjection)||(doing_EKT_now)){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->computes(d_lab->d_pressureExtraProjectionLabel);
    }else{
      tsk->modifies(d_lab->d_pressureExtraProjectionLabel);
    }
  }else {
    tsk->computes(timelabels->pressure_out);
    if (timelabels->recursion){
      tsk->computes(d_lab->d_InitNormLabel);
    }
  }


  sched->addTask(tsk, d_perproc_patches, matls);

  const Patch* d_pressRefPatch = 0;
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){

    const Patch* patch=*iter;
    if(patch->containsCell(d_pressRef)){
      d_pressRefPatch = patch;
    }
  }
  if(!d_pressRefPatch)
    throw InternalError("Patch containing pressure (and density) reference point was not found",
                        __FILE__, __LINE__);

  d_pressRefProc = lb->getPatchwiseProcessorAssignment(d_pressRefPatch);
}

//______________________________________________________________________
//
void 
PressureSolver::pressureLinearSolve_all(const ProcessorGroup* pg,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const TimeIntegratorLabel* timelabels,
                                        bool extraProjection,
                                        bool d_EKTCorrection,
                                        bool doing_EKT_now)
{
  int archIndex = 0; // only one arches material
  int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
  ArchesVariables pressureVars;
  int me = pg->myrank();
  // initializeMatrix...
  if ( d_construct_solver_obj && !d_always_construct_A )  {
    d_linearSolver->matrixCreate(d_perproc_patches, patches);
    d_construct_solver_obj = false; // deleted in the destructor
  } else {
    d_linearSolver->matrixCreate(d_perproc_patches, patches); 
  }
  for (int p = 0; p < patches->size(); p++) {
    const Patch *patch = patches->get(p);
    // This calls fillRows on linear(petsc) solver
    pressureLinearSolve(pg, patch, indx, old_dw, new_dw, pressureVars,
                        timelabels, extraProjection,
                        d_EKTCorrection, doing_EKT_now);
  }

#if 0  
  //__________________________________
  //debugging
  string desc = timelabels->integrator_step_name;
  int timestep = d_lab->d_sharedState->getCurrentTopLevelTimeStep(); 
  d_iteration ++; 
  d_linearSolver->print(desc,timestep,d_iteration);
#endif
  
  bool converged =  d_linearSolver->pressLinearSolve();
  if (converged) {
    for (int p = 0; p < patches->size(); p++) {
      const Patch *patch = patches->get(p);
      d_linearSolver->copyPressSoln(patch, &pressureVars);
    }
  } else {
    if (pg->myrank() == 0){
      cerr << "pressure solver not converged, using old values" << endl;
    }
    throw InternalError("pressure solver is diverging", __FILE__, __LINE__);
  }

  if ( !d_always_construct_A && d_construct_A ) {
    d_construct_A = false; 
    d_lab->recompile_taskgraph = true; 
  }
  
  double init_norm = d_linearSolver->getInitNorm();
  if (timelabels->recursion){
    new_dw->put(max_vartype(init_norm), d_lab->d_InitNormLabel);
  }
  
  if(d_pressRefProc == me){
    CCVariable<double> pressure;
    pressure.copyPointer(pressureVars.pressure);
    pressureVars.press_ref = pressure[d_pressRef];
    cerr << "press_ref for norm: " << pressureVars.press_ref << " " <<
      d_pressRefProc << endl;
  }
  MPI_Bcast(&pressureVars.press_ref, 1, MPI_DOUBLE, d_pressRefProc, pg->getComm());
  
  if (d_norm_pres){ 
    for (int p = 0; p < patches->size(); p++) {
      const Patch *patch = patches->get(p);
      normPressure(patch, &pressureVars);
      //    updatePressure(pg, patch, &pressureVars);
      // put back the results
    }
  }

  if (d_do_only_last_projection){
    if ((timelabels->integrator_step_name == "Predictor")||
        (timelabels->integrator_step_name == "Intermediate")) {
      pressureVars.pressure.initialize(0.0);
      if (pg->myrank() == 0){
        cout << "Projection skipped" << endl;
      }
    }else{ 
      if (!((timelabels->integrator_step_name == "Corrector")||
            (timelabels->integrator_step_name == "CorrectorRK3"))){
        throw InvalidValue("Projection can only be skipped for RK SSP methods",__FILE__, __LINE__); 
      }
    }
  }
  if ( d_always_construct_A ) { 
    d_linearSolver->destroyMatrix(); 
  }
}


//______________________________________________________________________
// Actual linear solve
void 
PressureSolver::pressureLinearSolve(const ProcessorGroup* pc,
                                    const Patch* patch,
                                    const int indx,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    ArchesVariables& pressureVars,
                                    const TimeIntegratorLabel* timelabels,
                                    bool extraProjection,
                                    bool d_EKTCorrection,
                                    bool doing_EKT_now)
{
  ArchesConstVariables constPressureVars;
  // Get the required data
  Ghost::GhostType  gn = Ghost::None;
  if ((extraProjection)||(doing_EKT_now)){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(pressureVars.pressure, d_lab->d_pressureExtraProjectionLabel,indx, patch);
    }else{
      new_dw->getModifiable(pressureVars.pressure, d_lab->d_pressureExtraProjectionLabel, indx, patch);
    }
  }else{
    new_dw->allocateAndPut(pressureVars.pressure, timelabels->pressure_out, indx, patch);
  }
  
  if (!((d_pressure_correction)||(extraProjection)||((d_EKTCorrection)&&(doing_EKT_now)))){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->copyOut(pressureVars.pressure, timelabels->pressure_guess, indx, patch);
    }else{
      new_dw->copyOut(pressureVars.pressure, timelabels->pressure_guess, indx, patch);
    }
  }else{
    pressureVars.pressure.initialize(0.0);
  }
  
  new_dw->get(constPressureVars.pressCoeff,       d_lab->d_presCoefPBLMLabel,      indx, patch, gn, 0);
  new_dw->get(constPressureVars.pressNonlinearSrc,d_lab->d_presNonLinSrcPBLMLabel, indx, patch, gn, 0);

  // for parallel code lisolve will become a recursive task and 
  // will make the following subroutine separate
  // get patch numer ***warning****
  // sets matrix
  if ( d_construct_A ) { 
    d_linearSolver->setMatrix(pc, patch, constPressureVars.pressCoeff); 
  }
  d_linearSolver->setRHS_X(pc, patch,pressureVars.pressure, constPressureVars.pressNonlinearSrc, d_construct_A);

}

// ************************************************************************
// Schedule addition of hydrostatic term to relative pressure calculated
// in pressure solve
// ************************************************************************
void
PressureSolver::sched_addHydrostaticTermtoPressure(SchedulerP& sched, 
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls,
                                                   const TimeIntegratorLabel* timelabels)

{
  Task* tsk = scinew Task("Psolve:addhydrostaticterm",
                          this, &PressureSolver::addHydrostaticTermtoPressure,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel,    gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn, 0 ); 

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_pressPlusHydroLabel);
  }else {
    tsk->modifies(d_lab->d_pressPlusHydroLabel);
  }

  sched->addTask(tsk, patches, matls);
}

// ****************************************************************************
// Actual addition of hydrostatic term to relative pressure
// This routine assumes that the location of the reference pressure is at (0.0,0.0,0.0)
// ****************************************************************************
void 
PressureSolver::addHydrostaticTermtoPressure(const ProcessorGroup*,
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

    constCCVariable<double> prel;
    CCVariable<double> pPlusHydro;
    constCCVariable<double> denMicro;
    constCCVariable<int> cellType;

    double gx = d_physicalConsts->getGravity(1);
    double gy = d_physicalConsts->getGravity(2);
    double gz = d_physicalConsts->getGravity(3);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(prel,     d_lab->d_pressurePSLabel,     indx, patch, gn, 0);
    old_dw->get(denMicro, d_lab->d_densityMicroLabel,   indx, patch, gn, 0);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       indx, patch, gn, 0);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel,indx, patch);
    }else{
      new_dw->getModifiable(pPlusHydro,  d_lab->d_pressPlusHydroLabel,indx, patch);
    }

    //__________________________________
    int mmwallid = d_boundaryCondition->getMMWallId();

    pPlusHydro.initialize(0.0);

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) { 
      IntVector c = *iter;
      double xx = cellinfo->xx[c.x()];
      double yy = cellinfo->yy[c.y()];
      double zz = cellinfo->zz[c.z()];
      if( cellType[c] != mmwallid){
        pPlusHydro[c] = prel[c] + denMicro[c] * (gx * xx + gy * yy + gz * zz);
      }
    }
  }
}

//****************************************************************************
// Pressure stencil weights
//****************************************************************************
void 
PressureSolver::calculatePressureCoeff(const Patch* patch,
                                       CellInformation* cellinfo,
                                       ArchesVariables* coeff_vars,
                                       ArchesConstVariables* constcoeff_vars)
{
  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) { 
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();

    IntVector E  = c + IntVector(1,0,0);   IntVector W  = c - IntVector(1,0,0); 
    IntVector N  = c + IntVector(0,1,0);   IntVector S  = c - IntVector(0,1,0);
    IntVector T  = c + IntVector(0,0,1);   IntVector B  = c - IntVector(0,0,1); 
  
    //__________________________________
    //compute areas
    double area_N  = cellinfo->sew[i] * cellinfo->stb[k];
    double area_S  = area_N;
    double area_EW = cellinfo->sns[j] * cellinfo->stb[k];
    double area_TB = cellinfo->sns[j] * cellinfo->sew[i];
    Stencil7& A = coeff_vars->pressCoeff[c];
    A.e = area_EW/(cellinfo->dxep[i]);
    A.w = area_EW/(cellinfo->dxpw[i]);
    A.n = area_N /(cellinfo->dynp[j]);
    A.s = area_S /(cellinfo->dyps[j]);
    A.t = area_TB/(cellinfo->dztp[k]);
    A.b = area_TB/(cellinfo->dzpb[k]);
  }

#ifdef divergenceconstraint
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  
  fort_prescoef_var(idxLo, idxHi, constcoeff_vars->density,
                    coeff_vars->pressCoeff[Arches::AE],
                    coeff_vars->pressCoeff[Arches::AW],
                    coeff_vars->pressCoeff[Arches::AN],
                    coeff_vars->pressCoeff[Arches::AS],
                    coeff_vars->pressCoeff[Arches::AT],
                    coeff_vars->pressCoeff[Arches::AB],
                    cellinfo->sew, cellinfo->sns, cellinfo->stb,
                    cellinfo->sewu, cellinfo->dxep, cellinfo->dxpw, 
                    cellinfo->snsv, cellinfo->dynp, cellinfo->dyps, 
                    cellinfo->stbw, cellinfo->dztp, cellinfo->dzpb);
#endif
}


//****************************************************************************
// Modify Pressure Stencil for Multimaterial
//****************************************************************************
void
PressureSolver::mmModifyPressureCoeffs(const Patch* patch,
                                       ArchesVariables* coeff_vars,
                                       ArchesConstVariables* constcoeff_vars)

{
  constCCVariable<double>& voidFrac = constcoeff_vars->voidFraction;

  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) { 
    IntVector c = *iter;
    Stencil7& A = coeff_vars->pressCoeff[c];
    
    IntVector E  = c + IntVector(1,0,0);   IntVector W  = c - IntVector(1,0,0); 
    IntVector N  = c + IntVector(0,1,0);   IntVector S  = c - IntVector(0,1,0);
    IntVector T  = c + IntVector(0,0,1);   IntVector B  = c - IntVector(0,0,1); 
  
    A.e *= 0.5 * (voidFrac[c] + voidFrac[E]);
    A.w *= 0.5 * (voidFrac[c] + voidFrac[W]);
    A.n *= 0.5 * (voidFrac[c] + voidFrac[N]);
    A.s *= 0.5 * (voidFrac[c] + voidFrac[S]);
    A.t *= 0.5 * (voidFrac[c] + voidFrac[T]);
    A.b *= 0.5 * (voidFrac[c] + voidFrac[B]);
  }
}


//______________________________________________________________________
//  
void 
PressureSolver::normPressure(const Patch* patch,
                             ArchesVariables* vars)
{
  double pressref = vars->press_ref;
 
  for(CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    vars->pressure[c] = vars->pressure[c] - pressref;
  } 
}  
//______________________________________________________________________

void 
PressureSolver::updatePressure(const Patch* patch,
                               ArchesVariables* vars)
{
  for(CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    vars->pressureNew[c] = vars->pressure[c];
  }
}  

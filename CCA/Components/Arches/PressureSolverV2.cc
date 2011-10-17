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

#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/PetscCommon.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PetscSolver.h>
#ifdef HAVE_HYPRE
#include <CCA/Components/Arches/HypreSolverV2.h>
#endif


#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>


using namespace Uintah;
using namespace std;

static DebugStream dbg("ARCHES_PRESS_SOLVE",false);

//______________________________________________________________________
// Default constructor for PressureSolver
//______________________________________________________________________
PressureSolver::PressureSolver(ArchesLabel* label,
                               const MPMArchesLabel* MAlb,
                               BoundaryCondition* bndry_cond,
                               PhysicalConstants* phys_const,
                               const ProcessorGroup* myworld,
                               SolverInterface* hypreSolver):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_boundaryCondition(bndry_cond),
                                     d_physicalConsts(phys_const),
                                     d_myworld(myworld),
                                     d_hypreSolver(hypreSolver)
{
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0; 
  d_iteration = 0;
  d_indx = -9;
  d_hypreSolver_parameters = NULL;
}

//______________________________________________________________________
// Destructor
//______________________________________________________________________
PressureSolver::~PressureSolver()
{
  // destroy A, x, and B

  if ( !d_always_construct_A ) { 
    d_linearSolver->destroyMatrix();
  }

  delete d_discretize;
  delete d_source;
  delete d_linearSolver;
}

//______________________________________________________________________
// Problem Setup
//______________________________________________________________________
void 
PressureSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("PressureSolver");
  d_pressRef = d_physicalConsts->getRefPoint();
  db->getWithDefault("normalize_pressure",      d_norm_press, false);
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
    d_linearSolver->problemSetup(db);
    d_whichSolver = petsc;
  }
#ifdef HAVE_HYPRE
  else if (linear_sol == "hypre"){
    d_whichSolver = hypre;
    d_hypreSolver_parameters = d_hypreSolver->readParameters(db, "pressure");
    d_hypreSolver_parameters->setSolveOnExtraCells(false);
    d_hypreSolver_parameters->setDynamicTolerance(true);
    
    d_linearSolver = scinew HypreSolver(d_myworld);
  }
#endif
  else {
    throw InvalidValue("Linear solver option"
                       " not supported" + linear_sol, __FILE__, __LINE__);
  }
}


//______________________________________________________________________
// Task that is called by Arches and constains scheduling of other tasks
//______________________________________________________________________
void PressureSolver::sched_solve(const LevelP& level,
                                 SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool extraProjection)
{

  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  lb->getPerProcessorPatchSet(level);
  
  int archIndex = 0; // only one arches material
  d_indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  string pressLabel = "NULL";
  
  sched_buildLinearMatrix( sched, perproc_patches, matls, 
                           timelabels, extraProjection); 
                           
  sched_setGuessForX(      sched, perproc_patches, matls,  
                           timelabels, extraProjection);
                        
  sched_setRHS_X_wrap(     sched, perproc_patches, matls,
                           timelabels, extraProjection);

  sched_SolveSystem(       sched, perproc_patches, matls, 
                           timelabels, extraProjection);


  schedExtract_X(          sched, perproc_patches, matls,
                           timelabels, extraProjection, pressLabel);
                 
  sched_normalizePress(    sched, perproc_patches, matls, pressLabel,timelabels);

  
  if ((d_MAlab)&&(!(extraProjection))) {
    sched_addHydrostaticTermtoPressure(sched, perproc_patches, matls, timelabels);
  }
}

//______________________________________________________________________
// Schedule build of linear matrix
//______________________________________________________________________
void 
PressureSolver::sched_buildLinearMatrix(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const TimeIntegratorLabel* timelabels,
                                        bool extraProjection)
{
  //  build pressure equation coefficients and source
  string taskname =  "PressureSolver::buildLinearMatrix_" +
                     timelabels->integrator_step_name;
  if (extraProjection){
    taskname += "_extraProjection";
  }
  
  printSchedule(patches,dbg,taskname);
  
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::buildLinearMatrix,
                          patches,
                          timelabels, extraProjection);
    

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
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel,       gn, 0);

  
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

  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First) && !extraProjection ) {
    tsk->computes(d_lab->d_presCoefPBLMLabel);
    tsk->computes(d_lab->d_presNonLinSrcPBLMLabel);
  } else {
    tsk->modifies(d_lab->d_presCoefPBLMLabel);
    tsk->modifies(d_lab->d_presNonLinSrcPBLMLabel);
  }
  

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
// Actually build of linear matrix for pressure equation
//______________________________________________________________________
void 
PressureSolver::buildLinearMatrix(const ProcessorGroup* pc,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /* matls */,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const PatchSet* patchSet,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection)
{
  // create matrix.
  bool isFirstTimestep = (d_lab->d_sharedState->getCurrentTopLevelTimeStep() == 1);

  if ( isFirstTimestep || d_always_construct_A )  {
    d_linearSolver->matrixCreate(patchSet, patches);
  }

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
    printTask(patches, patch,dbg,"buildLinearMatrix");
    
    ArchesVariables vars;
    ArchesConstVariables constVars;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;

    new_dw->get(constVars.cellType,     d_lab->d_cellTypeLabel,    d_indx, patch, gac, 1);
    new_dw->get(constVars.density,      d_lab->d_densityCPLabel,   d_indx, patch, gac, 1);
    new_dw->get(constVars.uVelRhoHat,   d_lab->d_uVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.vVelRhoHat,   d_lab->d_vVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.wVelRhoHat,   d_lab->d_wVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.filterdrhodt, d_lab->d_filterdrhodtLabel,d_indx, patch, gn,  0);
#ifdef divergenceconstraint
    new_dw->get(constVars.divergence,   d_lab->d_divConstraintLabel,d_indx, patch, gn, 0);
#endif

    if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First) && !extraProjection ) {
      new_dw->allocateAndPut(vars.pressCoeff,        d_lab->d_presCoefPBLMLabel,      d_indx, patch);
      new_dw->allocateAndPut(vars.pressNonlinearSrc, d_lab->d_presNonLinSrcPBLMLabel, d_indx, patch);
    }else{
      new_dw->getModifiable(vars.pressCoeff,         d_lab->d_presCoefPBLMLabel,      d_indx, patch);
      new_dw->getModifiable(vars.pressNonlinearSrc,  d_lab->d_presNonLinSrcPBLMLabel, d_indx, patch);
    }
    
    new_dw->allocateTemporary(vars.pressLinearSrc,  patch);
    vars.pressLinearSrc.initialize(0.0);   
    vars.pressNonlinearSrc.initialize(0.0);   

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, d_indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    //__________________________________
    calculatePressureCoeff(patch, cellinfo, &vars, &constVars);


    // Modify pressure coefficients for multimaterial formulation
    if (d_MAlab) {
      new_dw->get(constVars.voidFraction, d_lab->d_mmgasVolFracLabel, d_indx, patch,gac, 1);

      mmModifyPressureCoeffs(patch, &vars,  &constVars);

    }
    
    d_source->calculatePressureSourcePred(pc, patch, delta_t,
                                          cellinfo, &vars,
                                          &constVars);

    if (d_doMMS){
      d_source->calculatePressMMSSourcePred(pc, patch, delta_t,
                                            cellinfo, &vars,
                                            &constVars);
    }
    // do multimaterial bc; this is done before 
    // calculatePressDiagonal because unlike the outlet
    // boundaries in the explicit projection, we want to 
    // show the effect of AE, etc. in AP for the 
    // intrusion boundaries    
    if (d_MAlab){
      d_boundaryCondition->mmpressureBC(new_dw, patch,
                                        &vars, &constVars);
    }
    // Calculate Pressure Diagonal
    d_discretize->calculatePressDiagonal(patch, &vars);

    d_boundaryCondition->pressureBC(patch, d_indx, &vars, &constVars);
    
    // pass the coefficients into either hypre or petsc
    if ( d_construct_A ) { 
      d_linearSolver->setMatrix(pc, patch, vars.pressCoeff); 
    }
  }
}

//______________________________________________________________________
//  This task 
//______________________________________________________________________
void 
PressureSolver::sched_setGuessForX(SchedulerP& sched,
                                   const PatchSet* patches,                        
                                   const MaterialSet* matls,                       
                                   const TimeIntegratorLabel* timelabels,          
                                   bool extraProjection)                             
{ 

  string taskname =  "PressureSolver::sched_setGuessForX_" +
                     timelabels->integrator_step_name;
                     
  printSchedule(patches,dbg, taskname);

#if 0  
  cout << "     integrator_step_number:    " << timelabels->integrator_step_number << endl;
  cout << "     time integration name:     " << timelabels->integrator_step_name << endl;
  cout << "     timelabel->pressure_guess: " << timelabels->pressure_guess->getName() << endl;
  cout << "     timelabel->pressure_out:   " << timelabels->pressure_out->getName() << endl;
#endif
       
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::setGuessForX,
                          timelabels, extraProjection);
                          
  Ghost::GhostType  gn = Ghost::None;

  if (!extraProjection){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, timelabels->pressure_guess, gn, 0);
      tsk->computes( d_lab->d_pressureGuessLabel);
    }else{
      tsk->requires(Task::NewDW, timelabels->pressure_guess, gn, 0);
      tsk->modifies( d_lab->d_pressureGuessLabel );
    }
  }

  sched->addTask(tsk, patches, matls);
}



//______________________________________________________________________
//  setGuessForX
// This simply uses the previous iteration's or previous timestep's
// pressure as the initial guess for the pressure solver
//______________________________________________________________________
void 
PressureSolver::setGuessForX ( const ProcessorGroup* pg,
                               const PatchSubset* patches,                 
                               const MaterialSubset* matls,                      
                               DataWarehouse* old_dw,                      
                               DataWarehouse* new_dw,                      
                               const TimeIntegratorLabel* timelabels,      
                               const bool extraProjection )                  
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"setGuessForX");

    CCVariable<double> guess;
    CCVariable<double> press_old;
    
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->allocateTemporary(press_old, patch, gn, 0);
    
#if 0   
    cout << " time integration name:     " << timelabels->integrator_step_name << endl;
    cout << " timelabel->pressure_guess: " << timelabels->pressure_guess->getName() << endl;
    cout << " timelabel->pressure_out:   " << timelabels->pressure_out->getName() << endl;
#endif
    
    if ( !extraProjection ){
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        
        new_dw->allocateAndPut(guess,     d_lab->d_pressureGuessLabel, d_indx, patch);
        old_dw->copyOut(press_old,        timelabels->pressure_guess,  d_indx, patch);
        
        guess.copyData(press_old);
      }else{
        new_dw->getModifiable(guess,     d_lab->d_pressureGuessLabel,  d_indx, patch);
        new_dw->copyOut(press_old,      timelabels->pressure_guess,    d_indx, patch);
        
        guess.copyData(press_old);
      }
    }
    else{
      
      new_dw->allocateAndPut(guess,     d_lab->d_pressureGuessLabel, d_indx, patch);
      guess.initialize(0.0);
    }
  }
}



//______________________________________________________________________
// Schedule setPressRHS
//  This task just passes the uintah data through to either hypre or petsc
//______________________________________________________________________
void 
PressureSolver::sched_setRHS_X_wrap(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels,
                                    bool extraProjection)
{ 

  string taskname =  "PressureSolver::setRHS_X_wrap_" +
                     timelabels->integrator_step_name;
                     
  printSchedule(patches,dbg, taskname);
     
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::setRHS_X_wrap,
                          timelabels, extraProjection );

  Ghost::GhostType  gn = Ghost::None;
  
  tsk->requires(Task::NewDW, d_lab->d_pressureGuessLabel,    gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_presNonLinSrcPBLMLabel,gn, 0);

  sched->addTask(tsk, patches, matls);
}


//______________________________________________________________________
//  setRHS_X_wrap
//  This is a wrapper task and passes Uintah data to either Hypre or Petsc
//  which fills in the vector X and RHS
//______________________________________________________________________
void 
PressureSolver::setRHS_X_wrap ( const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const TimeIntegratorLabel* timelabels,
                                 const bool extraProjection )
{ 
  Ghost::GhostType  gn = Ghost::None;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"setRHS_X");
    
    constCCVariable<double> guess;
    constCCVariable<double> rhs;
     
    new_dw->get(guess, d_lab->d_pressureGuessLabel,     d_indx, patch, gn, 0);
    new_dw->get(rhs,   d_lab->d_presNonLinSrcPBLMLabel, d_indx, patch, gn, 0);
   
    // Pass the data to either Hypre or Petsc
    d_linearSolver->setRHS_X(pg, patch, guess, rhs, d_construct_A);
  }
  
  //__________________________________
  // set outputfile name
  if(d_whichSolver == hypre){
    string desc  = timelabels->integrator_step_name;
    int timestep = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    d_iteration ++;

    ostringstream fname;
    fname << "." << desc.c_str() << "." << timestep << "." << d_iteration;
    d_hypreSolver_parameters->setOutputFileName(fname.str());
  }
}




//______________________________________________________________________
// This task calls either Petsc or Hypre to solve the system
//______________________________________________________________________
void 
PressureSolver::sched_SolveSystem(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection)
{
  //__________________________________
  //  Hypre
  if ( d_whichSolver == hypre ) {
    const LevelP level = getLevelP(patches->getUnion());
    
    //__________________________________
    //  gross logic to determine what the
    //  solution label is and if it is a modified variable
    bool modifies_x = false;
    const VarLabel* pressLabel;
    
    if ( extraProjection ){
    
      pressLabel = d_lab->d_pressureExtraProjectionLabel;
      
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        modifies_x = false;
      }else{
        modifies_x = true;
      }
    } else {
      pressLabel = timelabels->pressure_out;
      modifies_x = false;
    }    
    
    const VarLabel* A     = d_lab->d_presCoefPBLMLabel;
    const VarLabel* x     = pressLabel;
    const VarLabel* b     = d_lab->d_presNonLinSrcPBLMLabel;
    const VarLabel* guess = d_lab->d_pressureGuessLabel;
    // cout << "guess Label " << guess->getName() << endl;
    
    d_hypreSolver->scheduleSolve(level, sched,  matls,
                                 A,      Task::NewDW,
                                 x,      modifies_x,
                                 b,      Task::NewDW,
                                 guess,  Task::NewDW,
                                 d_hypreSolver_parameters);
  }


  //__________________________________
  //  Petsc
  if( d_whichSolver == petsc ) {
    string taskname =  "PressureSolver::solveSystem_" + 
                       timelabels->integrator_step_name;

    if (extraProjection){
      taskname += "_extraProjection";
    }

    printSchedule(patches,dbg,taskname);

    Task* tsk = scinew Task(taskname, this,
                            &PressureSolver::solveSystem,
                            timelabels);

    if (timelabels->recursion){
      tsk->computes(d_lab->d_InitNormLabel);
    }

    sched->addTask(tsk, patches, matls);
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
void 
PressureSolver::solveSystem(const ProcessorGroup* pg,
                            const PatchSubset* perproc_patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const TimeIntegratorLabel* timelabels)
{
  printTask(dbg,"solveSystem");
 
  // Call Petsc to solve the system
  bool converged   =  d_linearSolver->pressLinearSolve();
  double init_norm = d_linearSolver->getInitNorm();
  
  //__________________________________
  //  debugging   
#if 0
  string desc  = timelabels->integrator_step_name;
  int timestep = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  d_iteration ++;
  
  cout << "//_______________________________STEP_" << desc << " " << timestep << " " << d_iteration << endl;
  
  d_linearSolver->print(desc,timestep,d_iteration );
#endif
  
  if (timelabels->recursion){
    new_dw->put(max_vartype(init_norm), d_lab->d_InitNormLabel);
  }  
  
  if (!converged) {
    proc0cout << "pressure solver not converged, using old values" << endl;
    throw InternalError("pressure solver is diverging", __FILE__, __LINE__);
  }

  if ( !d_always_construct_A && d_construct_A ) {
    d_construct_A = false; 
    d_lab->recompile_taskgraph = true; 
  }
}


//______________________________________________________________________
//  schedExtract_X:
//  This task places the solution to the into an array and sets the value
//  of the reference pressure.
//______________________________________________________________________
void 
PressureSolver::schedExtract_X(SchedulerP& sched,
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection,
                               string& pressLabelName)
{ 

  string taskname =  "PressureSolver::Extract_X_" +
                     timelabels->integrator_step_name;
  printSchedule(patches,dbg,taskname);
   
  WhichCM compute_or_modify = none;
  const VarLabel* pressLabel;
  if ( extraProjection ){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      pressLabel = d_lab->d_pressureExtraProjectionLabel;
      compute_or_modify = Computes;
    }else{
      pressLabel = d_lab->d_pressureExtraProjectionLabel;
      compute_or_modify = Modifies;
    }
  }else {
    pressLabel = timelabels->pressure_out;
    compute_or_modify = Computes;
    if (timelabels->recursion){
      pressLabel = d_lab->d_InitNormLabel;
    }
  }
  
  // The solution is always computed inside of the hypre solver
  if(d_whichSolver == hypre){ 
    compute_or_modify = Modifies;
  }
  
  
  pressLabelName = pressLabel->getName();
  const VarLabel* refPressLabel = timelabels->ref_pressure;
  
  const string integratorPhase = timelabels->integrator_step_name;
  
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::Extract_X, compute_or_modify, 
                          pressLabel, refPressLabel, integratorPhase);
                            
  if(compute_or_modify == Computes){
    tsk->computes(pressLabel);
  } else {
    tsk->modifies(pressLabel);
  }
   
  //__________________________________
  //  find the normalization pressure
  if (d_norm_press){ 
    tsk->computes(refPressLabel);
  }
    
  sched->addTask(tsk, patches, matls);
}


//______________________________________________________________________
//  Extract_X
//  This task places the solution to the into a uintah array and sets the value
//  of the reference pressure.
//______________________________________________________________________
void 
PressureSolver::Extract_X ( const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            WhichCM compute_or_modify,
                            const VarLabel* pressLabel,
                            const VarLabel* refPressLabel,
                            const string integratorPhase )
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"Extract_X " + integratorPhase);
    
    ArchesVariables vars;
    
    if(compute_or_modify == Computes){
      new_dw->allocateAndPut(vars.pressure,  pressLabel, d_indx, patch);
    } else {
      new_dw->getModifiable(vars.pressure,  pressLabel, d_indx, patch);
    }    
    
    d_linearSolver->copyPressSoln(patch, &vars);
    
    
    if ( d_always_construct_A ) {   // always destroy if you always construct
      d_linearSolver->destroyMatrix(); 
    }
    
    //__________________________________
    //  Find the reference pressure
    if (d_norm_press){ 
      double refPress = 0.0;
      if( patch->containsCell(d_pressRef)){
        refPress = vars.pressure[d_pressRef];        
      }
      new_dw->put(sum_vartype(refPress), refPressLabel);
    }
    
    if (d_do_only_last_projection){
      if ( integratorPhase == "Predictor" || integratorPhase == "Intermediate") {
        vars.pressure.initialize(0.0);

        proc0cout << "Projection skipped" << endl;
      }else{ 
        if (!(integratorPhase == "Corrector" || integratorPhase == "CorrectorRK3" ) ){
          throw InvalidValue("Projection can only be skipped for RK SSP methods",__FILE__, __LINE__); 
        }
      }
    }
    
    
  } // patches
}

//______________________________________________________________________
//  normalizePress:
//  Subtract off the reference pressure from pressure field
//______________________________________________________________________
void 
PressureSolver::sched_normalizePress(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const string& pressLabelname,
                                     const TimeIntegratorLabel* timelabels)
{
  // ignore this task if not normalizing
  if(!d_norm_press){
    return;
  }
  printSchedule(patches,dbg,"PressureSolver::normalizePress");
  
  const VarLabel* pressLabel    = VarLabel::find(pressLabelname);
  const VarLabel* refPressLabel = timelabels->ref_pressure;
  
  Task* tsk = scinew Task("PressureSolver::normalizePress",this, 
                          &PressureSolver::normalizePress, pressLabel, refPressLabel);                          

  tsk->modifies(pressLabel);
  tsk->requires(Task::NewDW, refPressLabel, Ghost::None, 0);
  
  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
//
//______________________________________________________________________
void 
PressureSolver::normalizePress ( const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw,
                                 const VarLabel* pressLabel,
                                 const VarLabel* refPressLabel)
{
  sum_vartype refPress = -9;
  new_dw->get(refPress, refPressLabel);
  
  proc0cout << "press_ref for norm: " << refPress << endl;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"NormalizePressure");
    
    CCVariable<double> press;
    new_dw->getModifiable(press,  pressLabel, d_indx, patch);

    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      press[c] = press[c] - refPress;
    } 
  }
}

//______________________________________________________________________
// Schedule addition of hydrostatic term to relative pressure calculated
// in pressure solve
//______________________________________________________________________
void
PressureSolver::sched_addHydrostaticTermtoPressure(SchedulerP& sched, 
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls,
                                                   const TimeIntegratorLabel* timelabels)

{
  printSchedule(patches,dbg,"PressureSolver::sched_addHydrostaticTermtoPressure");

  Task* tsk = scinew Task("PressureSolver::sched_addHydrostaticTermtoPressure",
                          this, &PressureSolver::addHydrostaticTermtoPressure,
                          timelabels);

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_pressurePSLabel,    gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_densityMicroLabel,  gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel,      gn, 0 ); 

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_pressPlusHydroLabel);
  }else {
    tsk->modifies(d_lab->d_pressPlusHydroLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
// Actual addition of hydrostatic term to relative pressure
// This routine assumes that the location of the reference pressure is at (0.0,0.0,0.0)
//______________________________________________________________________
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
    printTask(patches, patch,dbg,"addHydrostaticTermtoPressure");
    
    constCCVariable<double> prel;
    CCVariable<double> pPlusHydro;
    constCCVariable<double> denMicro;
    constCCVariable<int> cellType;

    double gx = d_physicalConsts->getGravity(1);
    double gy = d_physicalConsts->getGravity(2);
    double gz = d_physicalConsts->getGravity(3);
    
    int indx = d_lab->d_sharedState->getArchesMaterial(0)->getDWIndex();
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(prel,     d_lab->d_pressurePSLabel,     indx, patch, gn, 0);
    old_dw->get(denMicro, d_lab->d_densityMicroLabel,   indx, patch, gn, 0);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       indx, patch, gn, 0);
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel, indx, patch);
    }else{
      new_dw->getModifiable(pPlusHydro,  d_lab->d_pressPlusHydroLabel, indx, patch);
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

//______________________________________________________________________
// Pressure stencil weights
//______________________________________________________________________
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

  //__________________________________
  //  The petsc and hypre solvers need A not -A
  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) { 
    IntVector c = *iter;
    Stencil7& A = coeff_vars->pressCoeff[c];
    A.e *= -1.0; 
    A.w *= -1.0; 
    A.n *= -1.0; 
    A.s *= -1.0; 
    A.t *= -1.0; 
    A.b *= -1.0;
  }
}


//______________________________________________________________________
// Modify stencil weights (Pressure) to account for voidage due
// to multiple materials
//______________________________________________________________________
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

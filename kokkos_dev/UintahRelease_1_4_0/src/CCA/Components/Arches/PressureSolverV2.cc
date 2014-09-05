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
#include <CCA/Components/Arches/HypreSolver.h>
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
                               const ProcessorGroup* myworld):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_boundaryCondition(bndry_cond),
                                     d_physicalConsts(phys_const),
                                     d_myworld(myworld)
{
  d_discretize = 0;
  d_source = 0;
  d_linearSolver = 0; 
  d_construct_solver_obj = true; 
  d_iteration = 0;
}

//______________________________________________________________________
// Destructor
//______________________________________________________________________
PressureSolver::~PressureSolver()
{
  // destroy A, x, and B
  d_linearSolver->destroyMatrix();

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


//______________________________________________________________________
// Task that is called by Arches and constains scheduling of other tasks
//______________________________________________________________________
void PressureSolver::sched_solve(const LevelP& level,
                                 SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool extraProjection,
                                 bool d_EKTCorrection,
                                 bool doing_EKT_now)
{

  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  lb->getPerProcessorPatchSet(level);
  const PatchSubset* levelPatches = level->eachPatch()->getUnion();
  
  int archIndex = 0; // only one arches material
  d_indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
  const MaterialSet* matls = d_lab->d_sharedState->allArchesMaterials();
  string pressLabel = "NULL";
  
  if ( d_construct_solver_obj )  {
    d_linearSolver->matrixCreate(perproc_patches, levelPatches);
    d_construct_solver_obj = false; 
  }

  
  sched_buildLinearMatrix(sched, perproc_patches, matls, 
                          timelabels, extraProjection,d_EKTCorrection, doing_EKT_now);
                        
  sched_setRHS_X_wrap(sched, perproc_patches, matls,
                      timelabels, extraProjection,d_EKTCorrection, doing_EKT_now);

  sched_SolveSystem(sched, perproc_patches, matls, 
                    timelabels, extraProjection, doing_EKT_now);


  schedExtract_X(sched, perproc_patches, matls,
                 timelabels, extraProjection, doing_EKT_now,  pressLabel);
                 
  sched_normalizePress(sched, perproc_patches, matls, pressLabel);
  
  
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
                                        bool extraProjection,
                                        bool d_EKTCorrection,
                                        bool doing_EKT_now)
{
  //  build pressure equation coefficients and source
  string taskname =  "PressureSolver::buildLinearMatrix_" +
                     timelabels->integrator_step_name;
  if (extraProjection){
    taskname += "_extraProjection";
  }
  if (doing_EKT_now){
    taskname += "_EKTnow";
  }
  
  printSchedule(patches,dbg,taskname);
  
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
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel,       gn, 0);

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

//______________________________________________________________________
// Actually build of linear matrix for pressure equation
//______________________________________________________________________
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
    printTask(patches, patch,dbg,"buildLinearMatrix");
    
    ArchesVariables vars;
    ArchesConstVariables constVars;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->get(constVars.pressure, timelabels->pressure_guess, d_indx, patch, gn, 0);
    }else{
      new_dw->get(constVars.pressure, timelabels->pressure_guess, d_indx, patch, gn, 0);
    }
    new_dw->get(constVars.cellType,     d_lab->d_cellTypeLabel,    d_indx, patch, gac, 1);
    new_dw->get(constVars.density,      d_lab->d_densityCPLabel,   d_indx, patch, gac, 1);
    new_dw->get(constVars.uVelRhoHat,   d_lab->d_uVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.vVelRhoHat,   d_lab->d_vVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.wVelRhoHat,   d_lab->d_wVelRhoHatLabel,  d_indx, patch, gaf, 1);
    new_dw->get(constVars.filterdrhodt, d_lab->d_filterdrhodtLabel,d_indx, patch, gn,  0);
#ifdef divergenceconstraint
    new_dw->get(constVars.divergence,   d_lab->d_divConstraintLabel,d_indx, patch, gn, 0);
#endif

    if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        &&(((!(extraProjection))&&(!(d_EKTCorrection)))
           ||((d_EKTCorrection)&&(doing_EKT_now)))){
           

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
                                          &constVars,
                                          doing_EKT_now);

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
// Schedule setPressRHS
//  This task just passes the uintah data through to either hypre or petsc
//______________________________________________________________________
void 
PressureSolver::sched_setRHS_X_wrap(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels,
                                    bool extraProjection,
                                    bool d_EKTCorrection,
                                    bool doing_EKT_now)
{ 

  string taskname =  "PressureSolver::setRHS_X_wrap_" +
                     timelabels->integrator_step_name;
                     
  printSchedule(patches,dbg, taskname);
     
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::setRHS_X_wrap,
                          timelabels, extraProjection, d_EKTCorrection, doing_EKT_now);

  Ghost::GhostType  gn = Ghost::None;
  
/*`==========TESTING==========*/
  if (!((d_pressure_correction)||(extraProjection) ||((d_EKTCorrection)&&(doing_EKT_now)))){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      tsk->requires(Task::OldDW, timelabels->pressure_guess, gn, 0);
    }else{
      tsk->requires(Task::NewDW, timelabels->pressure_guess, gn, 0);
    }
  }
/*===========TESTING==========`*/
  
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
                                 const bool extraProjection,
                                 const bool d_EKTCorrection,
                                 const bool doing_EKT_now )
{ 
  Ghost::GhostType  gn = Ghost::None;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"setRHS_X");
    
    CCVariable<double> guess;
    constCCVariable<double> rhs;
    new_dw->allocateTemporary(guess, patch,gn,0);
    
    
/*`==========TESTING==========*/  
    if (!((d_pressure_correction)||(extraProjection)||((d_EKTCorrection)&&(doing_EKT_now)))){
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        old_dw->copyOut(guess, timelabels->pressure_guess, d_indx, patch);
      }
      else {
        new_dw->copyOut(guess, timelabels->pressure_guess, d_indx, patch);
      }
    }else{
      guess.initialize(0.0);
    } 
/*===========TESTING==========`*/

    new_dw->get(rhs,      d_lab->d_presNonLinSrcPBLMLabel, d_indx, patch, gn, 0);
   
    // Pass the data to either Hypre or Petsc
    d_linearSolver->setRHS_X(pg, patch, guess, rhs, d_construct_A);
  }
  //__________________________________
  //  debugging 
#if 1
  string desc  = timelabels->integrator_step_name;
  int timestep = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  d_iteration ++;
  
  cout << "//_______________________________STEP_" << desc << " " << timestep << " " << d_iteration << endl;
  
  d_linearSolver->print(desc,timestep,d_iteration );
#endif
}




//______________________________________________________________________
// This task calls either Petsc or Hypre to solve the system
//______________________________________________________________________
void 
PressureSolver::sched_SolveSystem(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection,
                                  bool doing_EKT_now)
{


  string taskname =  "PressureSolver::sched_SolveSystem_" + 
                     timelabels->integrator_step_name;
                     
  if (extraProjection){
    taskname += "_extraProjection";
  }
  if (doing_EKT_now){
    taskname += "_EKTnow";
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
 
  // Call Hypre or Petsc to solve the system
  bool converged   =  d_linearSolver->pressLinearSolve();
  double init_norm = d_linearSolver->getInitNorm();
  
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
                               bool doing_EKT_now,
                               string& pressLabelName)
{ 

  string taskname =  "PressureSolver::Extract_X_" +
                     timelabels->integrator_step_name;
  printSchedule(patches,dbg,taskname);
   
  WhichCM compute_or_modify = none;
  const VarLabel* pressLabel;
  if ((extraProjection)||(doing_EKT_now)){
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
  
  pressLabelName = pressLabel->getName();
  
  const string integratorPhase = timelabels->integrator_step_name;
  
  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::Extract_X, compute_or_modify, 
                          pressLabel, integratorPhase);
                            
  if(compute_or_modify == Computes){
    tsk->computes(pressLabel);
  } else {
    tsk->modifies(pressLabel);
  }
   
  //__________________________________
  //  find the normalization pressure
  if (d_norm_press){ 
    tsk->computes(d_lab->d_refPressure_label);
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
                            const VarLabel* varLabel,
                            const string integratorPhase )
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"Extract_X " + integratorPhase);
    
    ArchesVariables vars;
    
    if(compute_or_modify == Computes){
      new_dw->allocateAndPut(vars.pressure,  varLabel, d_indx, patch);
    } else {
      new_dw->getModifiable(vars.pressure,  varLabel, d_indx, patch);
    }    
    
    d_linearSolver->copyPressSoln(patch, &vars);
    
    //__________________________________
    //  Find the reference pressure
    if (d_norm_press){ 
      double refPress = 0.0;
      if( patch->containsCell(d_pressRef)){
        refPress = vars.pressure[d_pressRef];        
      }
      cout << " *** Extract X " << integratorPhase << " refPress " << refPress << endl;
      new_dw->put(sum_vartype(refPress), d_lab->d_refPressure_label);
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
                                     const string& pressLabelname)
{
  // ignore this task if not normalizing
  if(!d_norm_press){
    return;
  }
  printSchedule(patches,dbg,"PressureSolver::normalizePress");
  
  const VarLabel* pressLabel = VarLabel::find(pressLabelname);
  Task* tsk = scinew Task("PressureSolver::normalizePress",this, 
                          &PressureSolver::normalizePress, pressLabel);                          

  tsk->modifies(pressLabel);
  tsk->requires(Task::NewDW, d_lab->d_refPressure_label, Ghost::None, 0);
  
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
                                 const VarLabel* pressLabel)
{
  sum_vartype refPress = -9;
  new_dw->get(refPress, d_lab->d_refPressure_label);
  
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

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, d_indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(prel,     d_lab->d_pressurePSLabel,     d_indx, patch, gn, 0);
    old_dw->get(denMicro, d_lab->d_densityMicroLabel,   d_indx, patch, gn, 0);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       d_indx, patch, gn, 0);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(pPlusHydro, d_lab->d_pressPlusHydroLabel, d_indx, patch);
    }else{
      new_dw->getModifiable(pPlusHydro,  d_lab->d_pressPlusHydroLabel, d_indx, patch);
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

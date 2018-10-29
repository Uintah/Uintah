/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

//----- PressureSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
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
                               const ProcessorGroup* myworld,
                               SolverInterface* hypreSolver):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_boundaryCondition(bndry_cond),
                                     d_physicalConsts(phys_const),
                                     d_myworld(myworld),
                                     d_hypreSolver(hypreSolver)
{
  d_source = 0;
  d_iteration = 0;
  d_indx = -9;
  d_periodic_vector = IntVector(0,0,0);
}

//______________________________________________________________________
// Destructor
//______________________________________________________________________
PressureSolver::~PressureSolver()
{
  delete d_source;

  if(do_custom_arches_linear_solve){
    delete custom_solver;
  }
}

//______________________________________________________________________
// Problem Setup
//______________________________________________________________________
void
PressureSolver::problemSetup(ProblemSpecP& params,MaterialManagerP& materialManager)
{
  //do_custom_arches_linear_solve=true; //  Don't use hypre, use proto-type solver

  ProblemSpecP db = params->findBlock("PressureSolver");
  d_pressRef = d_physicalConsts->getRefPoint();
  db->getWithDefault("normalize_pressure",      d_norm_press, false);
  db->getWithDefault("do_only_last_projection", d_do_only_last_projection, false);

  // make source and boundary_condition objects
  d_source = scinew Source(d_physicalConsts, d_boundaryCondition);

  if(!do_custom_arches_linear_solve){
    d_hypreSolver->readParameters(db, "pressure" );

    d_hypreSolver->getParameters()->setSolveOnExtraCells(false);

    //force a zero setup frequency since nothing else
    //makes any sense at the moment.
    d_hypreSolver->getParameters()->setSetupFrequency(0.0);
  }

  d_enforceSolvability = false;
  if ( db->findBlock("enforce_solvability")){
    d_enforceSolvability = true;
  }

  //__________________________________
  // allow for addition of mass source terms
  if (db->findBlock("src")){
    string srcname;
    for( ProblemSpecP src_db = db->findBlock( "src" ); src_db != nullptr; src_db = src_db->findNextBlock( "src" ) ) {
      double weight;
      src_db->getAttribute("label", srcname);
      src_db->getWithDefault("weight", weight, 1.0);
      //which sources are turned on for this equation
      d_new_sources.push_back( srcname );
      d_source_weights.insert(std::make_pair(srcname, weight));
    }
  }

  //__________________________________
  // allow for addition of mass source terms with VarLabels
  if (db->findBlock("extra_src")){
    for( ProblemSpecP src_db = db->findBlock( "extra_src" ); src_db != nullptr; src_db = src_db->findNextBlock( "extra_src" ) ) {
      string srcname;
      src_db->getAttribute("label", srcname );

      const VarLabel * tempLabel;
      tempLabel = VarLabel::find( srcname );
      extraSourceLabels.push_back( tempLabel );

      // Note: varlabel must be registered prior to problem setup
      if (tempLabel == 0 ) {
        throw InvalidValue("Error: Cannot find the VarLabel for the source term: " + srcname, __FILE__, __LINE__);
      }
    }
  }
  nExtraSources = extraSourceLabels.size();

  //__________________________________
  // bulletproofing
  ProblemSpecP ps_root = params->getRootNode();
  ProblemSpecP sol_ps = ps_root->findBlock( "Solver" );
  string solver = "None";
  if( sol_ps ) {
    sol_ps->getAttribute( "type", solver );
  }
  if( !sol_ps || (solver != "hypre" && solver != "HypreSolver" && solver != "CGSolver") ){
    ostringstream msg;
    msg << "\n ERROR:Arches:PressureSolver  You've haven't specified the solver type.\n";
    msg << " Please add  <Solver type=\"hypre\" /> directly beneath <SimulationComponent type=\"arches\" /> \n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }
}

//______________________________________________________________________
//
void PressureSolver::scheduleInitialize( const LevelP& level,
                                         SchedulerP& sched,
                                         const MaterialSet* matls)
{
  if(do_custom_arches_linear_solve){
    int archIndex = 0; // only one arches material
    custom_solver = scinew linSolver(d_MAlab,   this, d_boundaryCondition ,d_physicalConsts ,d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(),
                                                                                             d_lab->d_mmgasVolFracLabel,
                                                                                             d_lab->d_cellTypeLabel,
                                                                                             d_lab->d_presCoefPBLMLabel,
                                                                                             d_lab->d_materialManager->allMaterials("Arches")
                                                                                                                       );
    custom_solver->sched_PreconditionerConstruction( sched, matls, level );// hard coded for level 0
  }else{
    d_hypreSolver->scheduleInitialize( level, sched, matls );
  }
}
//______________________________________________________________________
//
void PressureSolver::scheduleRestartInitialize( const LevelP& level,
                                                SchedulerP& sched,
                                                const MaterialSet* matls)
{
  d_hypreSolver->scheduleRestartInitialize( level, sched, matls );
}


//______________________________________________________________________
// Task that is called by Arches and constains scheduling of other tasks
//______________________________________________________________________
void PressureSolver::sched_solve(const LevelP& level,
                                 SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool extraProjection,
                                 const int rk_stage)
{

  d_periodic_vector = level->getPeriodicBoundaries();

  const PatchSet * perproc_patches =
    sched->getLoadBalancer()->getPerProcessorPatchSet(level);

  int archIndex = 0; // only one arches material
  d_indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
  const MaterialSet* matls = d_lab->d_materialManager->allMaterials( "Arches" );
  string pressLabel = "nullptr";

  sched_buildLinearMatrix( sched, perproc_patches, matls,
                           timelabels, extraProjection);

  sched_setGuessForX(      sched, perproc_patches, matls,
                           timelabels, extraProjection);

  sched_SolveSystem(       sched, perproc_patches, matls,
                           timelabels, extraProjection,
                           rk_stage );


  sched_set_BC_RefPress(   sched, perproc_patches, matls,
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

  tsk->requires(parent_old_dw, d_lab->d_delTLabel);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gac, 1);

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

  // add access to sources:
  for (auto iter = d_new_sources.begin(); iter != d_new_sources.end(); iter++){

    tsk->requires( Task::NewDW, VarLabel::find( *iter ), gn, 0 );

  }

  //extra sources
  for ( int i = 0; i < nExtraSources; i++ ) {
    tsk->requires( Task::NewDW, extraSourceLabels[i], gn, 0 );
  }

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
// Build the matrix and RHS for pressure equation
//  NOTE: you only need to create the matrix once.   We set it every timestep
//  because it's really difficult to turn it off inside the fortran code.
//  The majority of computational time is spent inside of the solver.
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
  DataWarehouse* parent_old_dw;
  if (timelabels->recursion){
    parent_old_dw = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  }else{
    parent_old_dw = old_dw;
  }

  delt_vartype delT;
  parent_old_dw->get(delT, d_lab->d_delTLabel );
  double delta_t = delT;
  delta_t *= timelabels->time_multiplier;

  Discretization* discrete = scinew Discretization(d_physicalConsts);

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

    //__________________________________
    calculatePressureCoeff(patch, &vars, &constVars);

    // Modify pressure coefficients for multimaterial formulation
    if (d_MAlab) {
      new_dw->get(constVars.voidFraction, d_lab->d_mmgasVolFracLabel, d_indx, patch,gac, 1);

      mmModifyPressureCoeffs(patch, &vars,  &constVars);

    }

    d_source->calculatePressureSourcePred(pc, patch, delta_t,
                                          &vars,
                                          &constVars,
                                          new_dw );

    Vector Dx = patch->dCell();
    double volume = Dx.x()*Dx.y()*Dx.z();

    // Add other source terms to the pressure:
    for ( auto iter = d_new_sources.begin(); iter != d_new_sources.end(); iter++){

      constCCVariable<double> src_value;
      new_dw->get( src_value, VarLabel::find( *iter ), d_indx, patch, gn, 0 );

      double weight = d_source_weights[*iter];

      // This may not be the most efficient way of adding the sources
      // to the RHS but in general we only expect 1 src to be added.
      // If the numbers of sources grow (>2), we may need to redo this.

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {

        IntVector c = *iter;
        vars.pressNonlinearSrc[c] += weight * src_value[c] / delta_t * volume;

      }
    }

    //-----ADD Extra Sources
    std::vector <constCCVariable<double> > extraSources (nExtraSources);
    for ( int i = 0; i < nExtraSources; i++ ) {
      const VarLabel* tempLabel = extraSourceLabels[i];
      new_dw->get( extraSources[i], tempLabel, d_indx, patch, gn, 0);
    }

    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      for ( int i = 0; i < nExtraSources; i++) {
        IntVector c = *iter;
        vars.pressNonlinearSrc[c] += extraSources[i][c] / delta_t * volume;
      }
    }

    // do multimaterial bc; this is done before
    // calculatePressDiagonal because unlike the outlet
    // boundaries in the explicit projection, we want to
    // show the effect of AE, etc. in AP for the
    // intrusion boundaries
    d_boundaryCondition->mmpressureBC(new_dw, patch,
                                      &vars, &constVars);

    // Calculate Pressure Diagonal
    discrete->calculatePressDiagonal(patch, &vars);

    d_boundaryCondition->pressureBC(patch, d_indx, &vars, &constVars);

  }
  delete discrete;
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

  tsk->requires( Task::OldDW, d_lab->d_timeStepLabel );

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

  //__________________________________
  // set outputfile name
  string desc  = timelabels->integrator_step_name;

  // int timeStep = d_lab->d_materialManager->getCurrentTopLevelTimeStep();
  timeStep_vartype timeStep;
  old_dw->get( timeStep, d_lab->d_timeStepLabel );

  d_iteration ++;

  ostringstream fname;
  fname << "." << desc.c_str() << "." << timeStep << "." << d_iteration;
  if(!do_custom_arches_linear_solve){
    d_hypreSolver->getParameters()->setOutputFileName(fname.str());
  }

}


//______________________________________________________________________
// This task calls UCF:hypre solver to solve the system
//______________________________________________________________________
void
PressureSolver::sched_SolveSystem(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls,
                                  const TimeIntegratorLabel* timelabels,
                                  bool extraProjection,
                                  const int rk_stage)
{
  const LevelP level = getLevelP(patches->getUnion());

  //__________________________________
  //  gross logic to determine what the
  //  solution label is and if it is a modified variable
  bool modifies_x   = false;
  bool isFirstSolve = true;
  const VarLabel* pressLabel;
  if (timelabels->integrator_step_number > 0)
    isFirstSolve = true;

  if ( extraProjection ){

    pressLabel = d_lab->d_pressureExtraProjectionLabel;

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      modifies_x   = false;
      isFirstSolve = true;
    }else{
      modifies_x   = true;
      isFirstSolve = false;
    }
  } else {
    pressLabel = timelabels->pressure_out;
    //    isFirstSolve = false;
    modifies_x = false;
  }

  const VarLabel* A     = d_lab->d_presCoefPBLMLabel;
  const VarLabel* x     = pressLabel;
  const VarLabel* b     = d_lab->d_presNonLinSrcPBLMLabel;
  const VarLabel* guess = d_lab->d_pressureGuessLabel;

  IntVector periodic_vector = level->getPeriodicBoundaries();
  const bool isPeriodic =periodic_vector.x() == 1 && periodic_vector.y() == 1 && periodic_vector.z() ==1;



  if(do_custom_arches_linear_solve){

    custom_solver->sched_customSolve(sched,  matls, patches,
                                     A,      Task::NewDW,
                                     x,      Task::NewDW,
                                     b,      Task::NewDW,
                                     guess,  Task::NewDW, rk_stage ,level);
  }else{
    if ( isPeriodic || d_enforceSolvability ) {
      d_hypreSolver->scheduleEnforceSolvability<CCVariable<double> >(level, sched, matls, b, rk_stage);
    }

    d_hypreSolver->scheduleSolve(level, sched,  matls,
                                 A,      Task::NewDW,
                                 x,      modifies_x,
                                 b,      Task::NewDW,
                                 guess,  Task::NewDW,
                                 isFirstSolve);
  }

  //add this?
  //if ( d_ref_value != 0. ) {
    //d_hypreSolver->scheduleSetReferenceValue<CCVariable<double> >( level, sched, matls, x, d_ref_loc );
  //}

}


//______________________________________________________________________
//  sched_set_BC_RefPress:
//  This task sets boundary conditions on the pressure and the value
//  of the reference pressure.
//______________________________________________________________________
void
PressureSolver::sched_set_BC_RefPress(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels,
                                      bool extraProjection,
                                      string& pressLabelName)
{

  string taskname =  "PressureSolver::set_BC_RefPress_" +
                     timelabels->integrator_step_name;

  printSchedule(patches,dbg,taskname);

  const VarLabel* pressLabel;
  if ( extraProjection ){
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      pressLabel = d_lab->d_pressureExtraProjectionLabel;
    }else{
      pressLabel = d_lab->d_pressureExtraProjectionLabel;
    }
  }else {
    pressLabel = timelabels->pressure_out;
  }


  pressLabelName = pressLabel->getName();
  const VarLabel* refPressLabel = timelabels->ref_pressure;

  const string integratorPhase = timelabels->integrator_step_name;

  Task* tsk = scinew Task(taskname, this,
                          &PressureSolver::set_BC_RefPress,
                          pressLabel, refPressLabel, integratorPhase);

  tsk->modifies(pressLabel);

  //__________________________________
  //  find the normalization pressure
  if (d_norm_press){
    tsk->computes(refPressLabel);
  }

  sched->addTask(tsk, patches, matls);
}


//______________________________________________________________________
//
//______________________________________________________________________
void
PressureSolver::set_BC_RefPress ( const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw,
                                  const VarLabel* pressLabel,
                                  const VarLabel* refPressLabel,
                                  const string integratorPhase )
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"set_BC_RefPress " + integratorPhase);

    ArchesVariables vars;
    new_dw->getModifiable(vars.pressure,  pressLabel, d_indx, patch);

    //__________________________________
    //  set boundary conditons on pressure
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

    for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;
      for(CellIterator iter=patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector c = *iter;
        vars.pressure[c] = 0;
      }
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

    int indx = d_lab->d_materialManager->getMaterial( "Arches", 0)->getDWIndex();

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
      Point P = patch->getCellPosition(c);
      double xx = P.x();
      double yy = P.y();
      double zz = P.z();
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
                                       ArchesVariables* coeff_vars,
                                       ArchesConstVariables* constcoeff_vars)
{

  Vector DX = patch->dCell();
  double area_EW = DX.y()*DX.z();
  double area_NS = DX.x()*DX.z();
  double area_TB = DX.x()*DX.y();

  for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
    IntVector c = *iter;

    IntVector E  = c + IntVector(1,0,0);   IntVector W  = c - IntVector(1,0,0);
    IntVector N  = c + IntVector(0,1,0);   IntVector S  = c - IntVector(0,1,0);
    IntVector T  = c + IntVector(0,0,1);   IntVector B  = c - IntVector(0,0,1);

    //__________________________________
    //compute areas
    Stencil7& A = coeff_vars->pressCoeff[c];
    A.e = area_EW/DX.x();
    A.w = area_EW/DX.x();
    A.n = area_NS/DX.y();
    A.s = area_NS/DX.y();
    A.t = area_TB/DX.z();
    A.b = area_TB/DX.z();
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

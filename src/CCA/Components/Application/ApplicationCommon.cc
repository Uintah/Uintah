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


#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <CCA/Ports/Output.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>

using namespace Uintah;

namespace {

Dout g_deltaT_warn_initial ( "DeltaTWarnInitial" , "ApplicationCommon", "Warn if the next delta T is greater than the initial maximum"              , true );
Dout g_deltaT_warn_increase( "DeltaTWarnIncrease", "ApplicationCommon", "Warn if the next delta T is increases more than a fraction of the previous", true );
Dout g_deltaT_warn_minimum ( "DeltaTWarnMinimum" , "ApplicationCommon", "Warn if the next delta T is less than the minimum"                         , true );
Dout g_deltaT_warn_maximum ( "DeltaTWarnMaximum" , "ApplicationCommon", "Warn if the next delta T is greater than the maximum"                      , true );
Dout g_deltaT_warn_clamp   ( "DeltaTWarnClamp"   , "ApplicationCommon", "Warn if the next delta T is clamped for output, checkpoint, or max time"   , true );

Dout g_deltaT_prevalidate    ( "DeltaTPreValidate"   , "ApplicationCommon", "Before reducing validate the next delta T w/warnings for each rank "        , false );
Dout g_deltaT_prevalidate_sum( "DeltaTPreValidateSum", "ApplicationCommon", "Before reducing validate the next delta T w/summary warning over all ranks ", false );

}


ApplicationCommon::ApplicationCommon( const ProcessorGroup   * myworld,
                                      const MaterialManagerP   materialManager )
    : UintahParallelComponent(myworld), m_materialManager(materialManager)
{
  // There should only be one MaterialManager. If there is a single
  // application the ComponentFactory will pass in a null pointer
  // which will trigger the MaterialManager to be created.

  // If there are multiple applications the Switcher (which is an
  // application) will create the MaterialManager and then pass that
  // to the child applications.

  // If there are combined applications (aka MPMICE) it will create
  // the MaterialManager and then pass that to the child applications.

  if (m_materialManager == nullptr) {
    m_materialManager = scinew MaterialManager();
  }

  //__________________________________
  //  These variables can be modified by an application.

  // Time Step
  m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription());

  // Simulation Time
  m_simulationTimeLabel = VarLabel::create(simTime_name, simTime_vartype::getTypeDescription());

  // delta t
  VarLabel* nonconstDelT = VarLabel::create(delT_name, delt_vartype::getTypeDescription());
  nonconstDelT->allowMultipleComputes();
  m_delTLabel = nonconstDelT;

  m_application_stats.calculateRankMinimum( true );
  m_application_stats.calculateRankStdDev ( true );

  // Reduction vars local to the application.
  
  // An application can request that an output or checkpoint been done
  // immediately.

  // output time step
  m_appReductionVars[ outputTimeStep_name ] = new
    ApplicationReductionVariable( outputTimeStep_name, bool_or_vartype::getTypeDescription() );

  // checkpoint time step
  m_appReductionVars[ checkpointTimeStep_name ] = new
    ApplicationReductionVariable( checkpointTimeStep_name, bool_or_vartype::getTypeDescription() );

  // An application may adjust the output interval or the checkpoint
  // interval during a simulation.  For example in deflagration ->
  // detonation simulations (Models/HEChem/DDT1.cc

  // output interval
  m_appReductionVars[ outputInterval_name ] = new
    ApplicationReductionVariable( outputInterval_name, min_vartype::getTypeDescription() );

  // checkpoint interval
  m_appReductionVars[ checkpointInterval_name ] = new
    ApplicationReductionVariable( checkpointInterval_name, min_vartype::getTypeDescription() );

  // An application may also request that the time step be recomputed,
  // aborted or the simulation end early.

  // Recompute the time step
  m_appReductionVars[ recomputeTimeStep_name ] = new
    ApplicationReductionVariable( recomputeTimeStep_name, bool_or_vartype::getTypeDescription() );

  // Abort the time step
  m_appReductionVars[ abortTimeStep_name ] = new
    ApplicationReductionVariable( abortTimeStep_name, bool_or_vartype::getTypeDescription() );

  // End the simulation
  m_appReductionVars[ endSimulation_name ] = new
    ApplicationReductionVariable( endSimulation_name, bool_or_vartype::getTypeDescription() );
}

ApplicationCommon::~ApplicationCommon()
{
  releaseComponents();
  
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
  VarLabel::destroy(m_delTLabel);

  for ( auto & var : m_appReductionVars ) {
    delete var.second;
  }
  
  m_appReductionVars.clear();
  
  // No need to delete the material manager as it is refcounted
  m_materialManager = nullptr;
}

void ApplicationCommon::setComponents( UintahParallelComponent *comp )
{
  ApplicationCommon * parent = dynamic_cast<ApplicationCommon*>( comp );

  attachPort( "scheduler",     parent->m_scheduler );
  attachPort( "load balancer", parent->m_loadBalancer );
  attachPort( "solver",        parent->m_solver );
  attachPort( "regridder",     parent->m_regridder );
  attachPort( "output",        parent->m_output );
  
  getComponents();
}

void ApplicationCommon::getComponents()
{
  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!", __FILE__, __LINE__);
  }

  m_loadBalancer = dynamic_cast<LoadBalancer*>( getPort("load balancer") );

  if( !m_loadBalancer ) {
    throw InternalError("dynamic_cast of 'm_loadBalancer' failed!", __FILE__, __LINE__);
  }

  m_solver = dynamic_cast<SolverInterface*>( getPort("solver") );

  if( !m_solver ) {
    throw InternalError("dynamic_cast of 'm_solver' failed!",  __FILE__, __LINE__);
  }

  m_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( isDynamicRegridding() && !m_regridder ) {
    throw InternalError("dynamic_cast of 'm_regridder' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }
}

void ApplicationCommon::releaseComponents()
{
  releasePort( "scheduler" );
  releasePort( "load balancer" );
  releasePort( "solver" );
  releasePort( "regridder" );
  releasePort( "output" );

  m_scheduler    = nullptr;
  m_loadBalancer = nullptr;
  m_solver       = nullptr;
  m_regridder    = nullptr;
  m_output       = nullptr;
}

void ApplicationCommon::problemSetup( const ProblemSpecP &prob_spec )
{
  // Check for an AMR attribute with the grid.
  ProblemSpecP grid_ps = prob_spec->findBlock("Grid");

  if (grid_ps) {
    grid_ps->getAttribute("doAMR", m_AMR);
    m_dynamicRegridding = m_AMR;
  }

  // If the AMR block is defined default to turning AMR on.
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");

  if (amr_ps) {
    m_AMR = true;

    std::string type;
    amr_ps->getAttribute("type", type);

    m_dynamicRegridding = (type.empty() || type == std::string("Dynamic"));

    amr_ps->get("useLockStep", m_lockstepAMR);
  }

  // Get the common time stepping specs
  ProblemSpecP time_ps = prob_spec->findBlock("Time");

  if (!time_ps) {
    throw ProblemSetupException("ERROR SimulationTime \n"
                                "Can not find the <Time> block.",
                                __FILE__, __LINE__);
  }

  // Sim time limits

  // Initial simulation time - will be written to the data warehouse
  // when SimulationController::timeStateSetup() is called.
  time_ps->require("initTime", m_simTime);

  // Maximum simulation time
  time_ps->require("maxTime", m_simTimeMax);

  // End the simulation at exactly the maximum simulation time
  if (!time_ps->get("end_at_max_time_exactly", m_simTimeEndAtMax)) {
    m_simTimeEndAtMax = false;
  }

  // Output time
  if (!time_ps->get("clamp_time_to_output", m_simTimeClampToOutput)) {
    m_simTimeClampToOutput = false;
  }

  // Time step limit
  if (!time_ps->get("max_Timesteps", m_timeStepsMax)) {
    m_timeStepsMax = 0;
  }

  // Wall time limit
  if (!time_ps->get("max_wall_time", m_wallTimeMax)) {
    m_wallTimeMax = 0;
  }

  // Delta T values - also used by the Switcher.
  problemSetupDeltaT( prob_spec );
}

void ApplicationCommon::problemSetupDeltaT( const ProblemSpecP &prob_spec )
{
  ProblemSpecP time_ps = prob_spec->findBlock("Time");

  if (!time_ps) {
    throw ProblemSetupException("ERROR SimulationTime \n"
                                "Can not find the <Time> block.",
                                __FILE__, __LINE__);
  }

  // Delta T limits
  ProblemSpecP tmp_ps;
  std::string flag;

  m_outputIfInvalidNextDelTFlag = 0;
  m_checkpointIfInvalidNextDelTFlag = 0;

  // When restarting use this delta T value
  if (!time_ps->get("override_restart_delt", m_delTOverrideRestart)) {
    m_delTOverrideRestart = 0.0;
  }

  // Multiply the next delta T value by this value
  time_ps->require("timestep_multiplier", m_delTMultiplier);

  // The maximum delta T can increase as a percent over the previous value
  if (!time_ps->get("max_delt_increase", m_delTMaxIncrease)) {
    m_delTMaxIncrease = 0;
  }
  else  // Can optionally output and/or checkpoint if exceeded
  {
    tmp_ps = time_ps->findBlock("max_delt_increase");
    tmp_ps->getAttribute("output", flag);
    if (flag == std::string("true")) {
      m_outputIfInvalidNextDelTFlag |= DELTA_T_MAX_INCREASE;
    }

    tmp_ps->getAttribute("checkpoint", flag);
    if (!flag.empty() && flag == std::string("true")) {
      m_checkpointIfInvalidNextDelTFlag |= DELTA_T_MAX_INCREASE;
    }
  }

  // The maximum delta T for the initial simulation time from
  // initial_delt_range
  if (!time_ps->get("delt_init", m_delTInitialMax)) {
    m_delTInitialMax = 0;
  }
  else  // Can optionally output and/or checkpoint if exceeded
  {
    tmp_ps = time_ps->findBlock("delt_init");
    tmp_ps->getAttribute("output", flag);
    if (flag == std::string("true")) {
      m_outputIfInvalidNextDelTFlag |= DELTA_T_INITIAL_MAX;
    }

    tmp_ps->getAttribute("checkpoint", flag);
    if (!flag.empty() && flag == std::string("true")) {
      m_checkpointIfInvalidNextDelTFlag |= DELTA_T_INITIAL_MAX;
    }
  }

  // The maximum simulation time which to enforce the delt_init
  if (!time_ps->get("initial_delt_range", m_delTInitialRange)) {
    m_delTInitialRange = 0;
  }

  // The minimum delta T value
  time_ps->require("delt_min", m_delTMin);
  // Can optionally output and/or checkpoint if exceeded
  tmp_ps = time_ps->findBlock("delt_min");
  tmp_ps->getAttribute("output", flag);
  if (flag == std::string("true")) {
    m_outputIfInvalidNextDelTFlag |= DELTA_T_MIN;
  }

  tmp_ps->getAttribute("checkpoint", flag);
  if (flag == std::string("true")) {
    m_checkpointIfInvalidNextDelTFlag |= DELTA_T_MIN;
  }

  // The maximum delta T value
  time_ps->require("delt_max", m_delTMax);
  // Can optionally output and/or checkpoint if exceeded
  tmp_ps = time_ps->findBlock("delt_max");
  tmp_ps->getAttribute("output", flag);
  if (flag == std::string("true")) {
    m_outputIfInvalidNextDelTFlag |= DELTA_T_MAX;
  }

  tmp_ps->getAttribute("checkpoint", flag);
  if (!flag.empty() && flag == std::string("true")) {
    m_checkpointIfInvalidNextDelTFlag |= DELTA_T_MAX;
  }
}


void
ApplicationCommon::scheduleTimeAdvance(const LevelP&,
                                             SchedulerP&)
{
  throw InternalError( "scheduleTimeAdvance is not implemented for this application", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleReduceSystemVars(const GridP& grid,
                                            const PatchSet* perProcPatchSet,
                                                  SchedulerP& scheduler)
{
  // Reduce the system vars which are on a per patch basis to a per
  // rank basis.
  Task* task = scinew Task("ApplicationCommon::reduceSystemVars", this,
                           &ApplicationCommon::reduceSystemVars);

  task->setType(Task::OncePerProc);
  task->usesMPI(true);

  // coarsen delT task requires that delT is computed on every level,
  // even if no tasks are run on that level.  I think this is a bug.
  // --Todd
  // TODO: Look into this - APH 02/27/17

  // Coarsen delT computes the global delT variable
  task->computes(m_delTLabel);

  for (int i = 0; i < grid->numLevels(); i++) {
    task->requires(Task::NewDW, m_delTLabel, grid->getLevel(i).get_rep());
  }
  
  // These are the application reduction variables. An application may
  // also request that the time step be recomputed, aborted, and/or the
  // simulation end early.

  // Check for a task computing the reduction variable, if found add
  // in a requires and activate the variable it will be tested.
  for (auto & var : m_appReductionVars) {
    const VarLabel* label = var.second->getLabel();
    
    if( scheduler->getComputedVars().find( label ) != scheduler->getComputedVars().end() ) {
      activateReductionVariable(var.first, true);
      
      task->requires(Task::NewDW, label);
      task->computes(label);
    }
  }

  // These two reduction vars may be set by the application via a
  // compute in which case a requires is needed (done above). Or if
  // the flag is set by ApplicationCommon, no requires is needed but
  // the reduction var needs to be active for the reduction and
  // subsequent test.
  if( m_outputIfInvalidNextDelTFlag ) {
    activateReductionVariable(outputTimeStep_name, true);
  }

  if( m_checkpointIfInvalidNextDelTFlag ) {
    activateReductionVariable(checkpointTimeStep_name, true);
  }

  // The above three tasks are on a per proc basis any rank can make
  // the request because it is a either benign or a set value.
  scheduler->addTask(task, perProcPatchSet, m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::reduceSystemVars( const ProcessorGroup * pg,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * matls,
                                           DataWarehouse  * old_dw,
                                           DataWarehouse  * new_dw )
{
  ValidateFlag validDelT = 0;

  // The goal of this task is to line up the delT across all levels.
  // If the coarse delT already exists (the one without an associated
  // level), then the application is not doing AMR.
  Patch* patch = nullptr;

  if (patches->size() != 0 && !new_dw->exists(m_delTLabel, -1, patch)) {
    // Start with the time step multiplier.
    double multiplier = m_delTMultiplier;

    const GridP grid = patches->get(0)->getLevel()->getGrid();

    for (int l = 0; l < grid->numLevels(); l++) {
      const LevelP level = grid->getLevel(l);

      if (l > 0 && !m_lockstepAMR) {
        multiplier *= level->getRefinementRatioMaxDim();
      }

      if (new_dw->exists(m_delTLabel, -1, *level->patchesBegin())) {
        delt_vartype delTvar;
        new_dw->get(delTvar, m_delTLabel, level.get_rep());

        // Adjust the local next delT by the multiplier
        m_delTNext = delTvar * multiplier;

        // Valiadate before the reduction. This assures that there will
        // not be any possible round off error for the next delta T.
        if (g_deltaT_prevalidate || g_deltaT_prevalidate_sum) {
          validDelT = validateNextDelT(m_delTNext, l);
        }

        new_dw->put(delt_vartype(m_delTNext), m_delTLabel);
      }

      // What should happen if there is no delta T???
    }
  }

  if (d_myworld->nRanks() > 1) {
    new_dw->reduceMPI(m_delTLabel, 0, 0, -1);
  }

  // Get the reduced next delta T
  delt_vartype delTvar;
  new_dw->get(delTvar, m_delTLabel);
  m_delTNext = delTvar;

  // Validate after the reduction. NOTE: Because each rank will
  // independently modify delta T the resulting values may be
  // different due to round off.
  if (!g_deltaT_prevalidate && !g_deltaT_prevalidate_sum) {
    // Validate and put the value into the warehouse if it changed.
    if ((validDelT = validateNextDelT(m_delTNext, -1))) {
      new_dw->override(delt_vartype(m_delTNext), m_delTLabel);
    }
  }

  // If delta T has been changed and if requested, for that change
  // output or checkpoint. Must be done before the reduction call.
  if (validDelT & m_outputIfInvalidNextDelTFlag) {
    setReductionVariable(new_dw, outputTimeStep_name, true);
  }

  if (validDelT & m_checkpointIfInvalidNextDelTFlag) {
    setReductionVariable(new_dw, checkpointTimeStep_name, true);
  }

  // Reduce the application specific reduction variables. If no value
  // was computed on an MPI rank, a benign value will be set. If the
  // reduction result is also a benign value, that means no MPI rank
  // wants to change the value and it will be ignored.
  for (auto & var : m_appReductionVars) {
    var.second->reduce(new_dw);
  }

  // When checking a reduction var, if it is not a benign value then
  // it was set at some point by at least one rank. Which is the only
  // time the value should be use.

  // Specific handling for reduction vars that need the grid.
  if (patches->size() != 0) {
    const GridP grid = patches->get(0)->getLevel()->getGrid();

    if (!isBenignReductionVariable(outputTimeStep_name)) {
      m_output->setOutputTimeStep(true, grid);
    }

    if (!isBenignReductionVariable(checkpointTimeStep_name)) {
      m_output->setCheckpointTimeStep(true, grid);
    }
  }

  // Specific handling for other reduction vars.
  if (!isBenignReductionVariable(outputInterval_name)) {
    m_output->setOutputInterval(getReductionVariable(outputInterval_name));
  }

  if (!isBenignReductionVariable(checkpointInterval_name)) {
    m_output->setCheckpointInterval(getReductionVariable(checkpointInterval_name));
  }

  checkReductionVars( pg, patches, matls, old_dw, new_dw );
  
}  // end reduceSysVar()


//______________________________________________________________________
//
void
ApplicationCommon::scheduleInitializeSystemVars( const GridP      & grid,
                                                 const PatchSet   * perProcPatchSet,
                                                       SchedulerP & scheduler)
{
  // Initialize the system vars which are on a per rank basis.
  Task* task = scinew Task("ApplicationCommon::initializeSystemVars", this,
                           &ApplicationCommon::initializeSystemVars);

  task->setType(Task::OncePerProc);
  
  task->computes(m_timeStepLabel);
  task->computes(m_simulationTimeLabel);

  // treatAsOld copyData noScrub notCopyData noCheckpoint
  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(), false, false, false, true, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(), false, false, false, true, true);

  scheduler->addTask(task, perProcPatchSet, m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::initializeSystemVars( const ProcessorGroup *,
                                         const PatchSubset    * patches,
                                         const MaterialSubset * /*matls*/,
                                               DataWarehouse  * /*old_dw*/,
                                               DataWarehouse  * new_dw )
{
  // Initialize the time step.
  new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

  // Initialize the simulation time.
  new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);
}


//______________________________________________________________________
//
void
ApplicationCommon::scheduleUpdateSystemVars(const GridP& grid,
                                            const PatchSet* perProcPatchSet,
                                                  SchedulerP& scheduler)
{
  // Update the system vars which are on a per rank basis.
  Task* task = scinew Task("ApplicationCommon::updateSystemVars", this,
                           &ApplicationCommon::updateSystemVars);

  task->setType(Task::OncePerProc);
  
  task->computes(m_timeStepLabel);
  task->computes(m_simulationTimeLabel);

  // treatAsOld copyData noScrub notCopyData noCheckpoint
  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(), false, false, false, true, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(), false, false, false, true, true);
    
  scheduler->addTask(task, perProcPatchSet, m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::updateSystemVars( const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * /*matls*/,
                                           DataWarehouse  * /*old_dw*/,
                                           DataWarehouse  * new_dw )
{  
  // If recomputing a time step do not update the time step or the simulation time.
  if ( !getReductionVariable( recomputeTimeStep_name ) ) {
    // Store the time step so it can be incremented at the top of the
    // time step where it is over written.
    new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);
    
    // Update the simulation time.
    m_simTime += m_delT;

    // Question - before putting the value into the warehouse should
    // it be broadcasted to assure it is sync'd acrosss all ranks?
    // Uintah::MPI::Bcast( &m_simTime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

    new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);
  }
}

//______________________________________________________________________
//
void
ApplicationCommon::scheduleRefine(const PatchSet*, SchedulerP&)
{
  throw InternalError( "scheduleRefine not implemented for this application\n", __FILE__, __LINE__ );
}

//______________________________________________________________________
//
void
ApplicationCommon::scheduleRefineInterface(const LevelP&,
                                                 SchedulerP&,
                                                 bool, bool)
{
  throw InternalError( "scheduleRefineInterface is not implemented for this application\n", __FILE__, __LINE__ );
}

//______________________________________________________________________
//
void
ApplicationCommon::scheduleCoarsen(const LevelP&,
                                         SchedulerP&)
{
  throw InternalError( "scheduleCoarsen is not implemented for this application\n", __FILE__, __LINE__ );
}

//______________________________________________________________________
//
void
ApplicationCommon::scheduleErrorEstimate( const LevelP&,
                                                SchedulerP& )
{
  throw InternalError( "scheduleErrorEstimate is not implemented for this application", __FILE__, __LINE__ );
}

//______________________________________________________________________
//
void
ApplicationCommon::scheduleInitialErrorEstimate(const LevelP& /*coarseLevel*/,
                                                      SchedulerP& /*sched*/)
{
  throw InternalError("scheduleInitialErrorEstimate is not implemented for this application", __FILE__, __LINE__);
}

//______________________________________________________________________
//
double
ApplicationCommon::getSubCycleProgress(DataWarehouse* fineDW)
{
  // DWs are always created in order of time.
  int fineID = fineDW->getID();  
  int coarseNewID = fineDW->getOtherDataWarehouse(Task::CoarseNewDW)->getID();

  // Need to do this check, on init timestep, old DW is nullptr, and
  // getOtherDW will throw exception.
  if (fineID == coarseNewID) {
    return 1.0;
  }

  int coarseOldID = fineDW->getOtherDataWarehouse(Task::CoarseOldDW)->getID();
  
  return ((double)fineID-coarseOldID) / (coarseNewID-coarseOldID);
}

//______________________________________________________________________
//
void
ApplicationCommon::recomputeDelT()
{
  // Get the new delT from the actual application.

  // Call the actual application method which if defined overrides the
  // virtual default method for the delta T
  double new_delT = recomputeDelT(m_delT);

  proc0cout << "WARNING Recomputng time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ", changing delT from " << m_delT
            << " to " << new_delT
            << std::endl;

  // Bulletproofing
  if (new_delT < m_delTMin || new_delT <= 0) {
    std::ostringstream warn;
    warn << "The new delT (" << new_delT << ") is either less than " << "minDelT (" << m_delTMin << ") or equal to 0";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }

  // When recomputing the delT, rank 0 determines the value and
  // sends it to all other ranks.
  Uintah::MPI::Bcast(&new_delT, 1, MPI_DOUBLE, 0, d_myworld->getComm());

  m_delT = new_delT;
}

//______________________________________________________________________
//
double
ApplicationCommon::recomputeDelT(const double delT)
{
  throw InternalError("recomputeDelT is not implemented for this application", __FILE__, __LINE__);
}

//______________________________________________________________________
//
void
ApplicationCommon::prepareForNextTimeStep()
{
  // Increment (by one) the current time step number so components know
  // what time step they are on and get the delta T that will be used.
  incrementTimeStep();

  // Get the delta that will be used for the time step.
  delt_vartype delt_var;
  m_scheduler->getLastDW()->get( delt_var, m_delTLabel );
  m_delT = delt_var;

  // Clear the time step based reduction variables.
  for ( auto & var : m_appReductionVars ) {
    var.second->reset();
  }
}

//______________________________________________________________________
//
void
ApplicationCommon::setDelTForAllLevels(       SchedulerP& scheduler,
                                        const GridP & grid,
                                        const int totalFine )
{
  // Adjust the delT for each level and store it in all applicable dws.
  double delT_fine = m_delT;
  int skip = totalFine;

  for (int i = 0; i < grid->numLevels(); ++i) {
    const Level* level = grid->getLevel(i).get_rep();

    if (isAMR() && i != 0 && !isLockstepAMR()) {
      int trr = level->getRefinementRatioMaxDim();
      delT_fine /= trr;
      skip /= trr;
    }

    for (int idw = 0; idw < totalFine; idw += skip) {
      DataWarehouse* dw = scheduler->get_dw(idw);
      dw->override(delt_vartype(delT_fine), m_delTLabel, level);

      // In a similar fashion write the time step and simulation time
      // to all DWs when running AMR grids.
      dw->override(timeStep_vartype(m_timeStep), m_timeStepLabel);

      dw->override(simTime_vartype(m_simTime), m_simulationTimeLabel);
    }
  }

  // Override for the global level as well (only matters on dw 0)
  DataWarehouse* oldDW = scheduler->get_dw(0);
  oldDW->override(delt_vartype(delT_fine), m_delTLabel);
}

//______________________________________________________________________
//
// This method is called only at restart - see
// SimulationController::timeStateSetup()  or by the in situ - see
// visit_DeltaTVariableCallback().

void
ApplicationCommon::setNextDelT( double delT, bool restart )
{
  // Restart - Check to see if the user has set a restart delT.
  if (restart && m_delTOverrideRestart) {
    proc0cout << "Overriding restart delT " << m_delT << " with " << m_delTOverrideRestart << "\n";

    m_delTNext = m_delTOverrideRestart;

    m_scheduler->getLastDW()->override(delt_vartype(m_delTNext), m_delTLabel);
  }

  // Restart - Otherwise get the next delta T from the archive.
  else if (restart && m_scheduler->getLastDW()->exists(m_delTLabel)) {
    delt_vartype delt_var;
    m_scheduler->getLastDW()->get(delt_var, m_delTLabel);
    m_delTNext = delt_var;
  }

  // All else fails use the delta T passed in. If from a restart it
  // would be the value used at the last time step. 
  else {
    m_delTNext = delT;
    m_scheduler->getLastDW()->override(delt_vartype(m_delTNext), m_delTLabel);
  }
}

//______________________________________________________________________
//
ValidateFlag
ApplicationCommon::validateNextDelT( double & delTNext, unsigned int level )
{
  // NOTE: This check is performed BEFORE the simulation time is
  // updated. As such, being that the time step has completed, the
  // actual simulation time is the current simulation time plus the
  // current delta T.

  // The invalid flag is a bitwise XOR for the local rank. Each bit
  // represents which threshold was exceeded. It is reduced on all
  // ranks if the pre-validating min flag is true (and the
  // pre-validating maxflag is false).
  std::ostringstream header, message;

  header << "WARNING ";

  // For the pre-validate report the rank and level.
  if (g_deltaT_prevalidate)
    header << "Rank-" << d_myworld->myRank() << " for level " << level << " ";

  header << "at time step " << m_timeStep << " and sim time " << m_simTime + m_delT << " : ";

  ValidateFlag invalid = 0;

  // Check to see if the next delT was increased too much over the
  // current delT
  double delt_tmp = (1.0 + m_delTMaxIncrease) * m_delT;

  if (m_delTMaxIncrease > 0 && delt_tmp > 0 && delTNext > delt_tmp) {
    invalid |= DELTA_T_MAX_INCREASE;

    if (g_deltaT_warn_increase) {
      if (!message.str().empty())
        message << std::endl;

      message << header.str()
              << "lowering the next delT from " << delTNext
              << " to the maximum: " << delt_tmp
              << " (maximum increase permitted is "
              << 1.0 + m_delTMaxIncrease << "x the previous)";
    }

    delTNext = delt_tmp;
  }

  // Check to see if the next delT is below the minimum delt
  if (m_delTMin > 0 && delTNext < m_delTMin) {
    invalid |= DELTA_T_MIN;

    if (g_deltaT_warn_minimum) {
      if (!message.str().empty()) {
        message << std::endl;
      }

      message << header.str() << "raising the next delT from " << delTNext << " to the minimum: " << m_delTMin;
    }

    delTNext = m_delTMin;
  }

  // Check to see if the next delT exceeds the maximum delt
  if (m_delTMax > 0 && delTNext > m_delTMax) {
    invalid |= DELTA_T_MAX;

    if (g_deltaT_warn_maximum) {
      if (!message.str().empty()) {
        message << std::endl;
      }

      message << header.str() << "lowering the next delT from " << delTNext << " to the maximum: " << m_delTMax;
    }

    delTNext = m_delTMax;
  }

  // Check to see if the next delT exceeds the maximum initial delt
  // This check shoud be last because it is for the initial time steps.
  if (m_delTInitialMax > 0 && m_simTime + m_delT <= m_delTInitialRange && delTNext > m_delTInitialMax) {
    invalid |= DELTA_T_INITIAL_MAX;

    if (g_deltaT_warn_initial) {
      if (!message.str().empty()) {
        message << std::endl;
      }

      message << header.str()
              << "for the initial time up to " << m_delTInitialRange
              << " lowering the next delT from " << delTNext
              << " to the maximum: " << m_delTInitialMax;
    }

    delTNext = m_delTInitialMax;
  }

  // Perform last so to possibly not cause other checks and warnings
  // as these checks may reduce the delta T to be smaller than the
  // minimums. Unless requested, no warning is issued as there is no
  // problem with the next delta T.

  // Adjust the next delT to clamp the simulation time to the requested
  // output and/or checkpoint times.
  if (m_simTimeClampToOutput) {

    // Adjust the next delta T to clamp the simulation time to the
    // output time.
    double nextOutput = m_output->getNextOutputTime();
    if (!m_output->isOutputTimeStep() && nextOutput != 0 && m_simTime + m_delT + delTNext > nextOutput) {
      invalid |= CLAMP_TIME_TO_OUTPUT;

      if (g_deltaT_warn_clamp) {
        if (!message.str().empty())
          message << std::endl;

        message << header.str()
                << "lowering the next delT from " << delTNext
                << " to " << nextOutput - (m_simTime + m_delT)
                << " to line up with the next output time: " << nextOutput;
      }

      delTNext = nextOutput - (m_simTime + m_delT);
    }

    // Adjust the next delta T to clamp the simulation time to the
    // checkpoint time.
    double nextCheckpoint = m_output->getNextCheckpointTime();
    if (!m_output->isCheckpointTimeStep() && nextCheckpoint != 0 && m_simTime + m_delT + delTNext > nextCheckpoint) {
      invalid |= CLAMP_TIME_TO_CHECKPOINT;

      if (g_deltaT_warn_clamp) {
        if (!message.str().empty()) {
          message << std::endl;
        }

        message << header.str()
                << "lowering the next delT from " << delTNext
                << " to " << nextCheckpoint - (m_simTime + m_delT)
                << " to line up with the next checkpoint time: " << nextCheckpoint;
      }

      delTNext = nextCheckpoint - (m_simTime + m_delT);
    }
  }

  // Adjust delta T so to end at the max simulation time.
  if (m_simTimeEndAtMax && m_simTime + m_delT + delTNext > m_simTimeMax) {
    invalid |= CLAMP_TIME_TO_MAX;

    if (g_deltaT_warn_clamp) {
      if (!message.str().empty())
        message << std::endl;

      message << header.str()
              << "lowering the next delT from " << delTNext
              << " to " << m_simTimeMax - (m_simTime + m_delT)
              << " to line up with the maximum simulation time of " << m_simTimeMax;
    }

    delTNext = m_simTimeMax - (m_simTime + m_delT);
  }

  // Check for a message which indicates that delta T was adjusted and
  // the user wants to be warned (see the g_deltaT_major_warnings and
  // g_deltaT_minor_warnings flags).
  if (!message.str().empty()) {
    // The pre-validate flag is true but not the pre-validate sum flag
    // report for all ranks or if no pre-validating flags are set
    // then a post-validate (default) so reprort for rank 0 only.
    if ((g_deltaT_prevalidate && !g_deltaT_prevalidate_sum) || (!g_deltaT_prevalidate && !g_deltaT_prevalidate_sum
                                                                && d_myworld->myRank() == 0)) {
      DOUT(true, message.str());
    }
  }

  // Report if pre-validating sum flag is set and only for level zero
  // as it is a summary.
  if (g_deltaT_prevalidate_sum && level == 0) {
    // Gather all of the bits where the threshold was exceeded.
    ValidateFlag invalidAll;

    Uintah::MPI::Reduce(&invalid, &invalidAll, 1, MPI_UNSIGNED_CHAR, MPI_BOR, 0, d_myworld->getComm());

    // Only report the summary on rank 0. One line for each instance
    // where the threshold was exceeded.
    if (d_myworld->myRank() == 0) {
      std::ostringstream header;
      header << "WARNING " << "at time step " << m_timeStep << " " << "and sim time " << m_simTime + m_delT << " : ";

      std::ostringstream message;

      // Report the warnings
      if (g_deltaT_warn_increase && (invalidAll & DELTA_T_MAX_INCREASE)) {
        if (!message.str().empty()) {
          message << std::endl;
        }

        message << header.str()
                << "for one or more ranks the next delta T was lowered."
                << " The maximum increase permitted is "
                << 1.0 + m_delTMaxIncrease << "x the previous";
      }

      if (g_deltaT_warn_minimum && (invalidAll & DELTA_T_MIN)) {
        if (!message.str().empty())
          message << std::endl;

        message << header.str() << "for one or more ranks the next delta T was " << "raised to the minimum: " << m_delTMin;
      }

      if (g_deltaT_warn_maximum && (invalidAll & DELTA_T_MAX)) {
        if (!message.str().empty()) {
          message << std::endl;
        }

        message << header.str() << "for one or more ranks the next delta T was " << "lowered to the maximum: " << m_delTMax;
      }

      if (g_deltaT_warn_initial && (invalidAll & DELTA_T_INITIAL_MAX)) {
        if (!message.str().empty()) {
          message << std::endl;
        }

        message << header.str() << "for one or more ranks " << "for the initial time up to " << m_delTInitialRange
                << " the next delT was lowered to " << m_delTInitialMax;
      }

      if (g_deltaT_warn_clamp) {
        if (invalidAll & CLAMP_TIME_TO_OUTPUT) {
          if (!message.str().empty()) {
            message << std::endl;
          }

          message << header.str() << "for one or more ranks the next delta T was "
                  << "lowered to line up with the next output time: " << m_output->getNextOutputTime();
        }

        if (invalidAll & CLAMP_TIME_TO_CHECKPOINT) {
          if (!message.str().empty()) {
            message << std::endl;
          }

          message << header.str()
                  << "for one or more ranks the next delta T was "
                  << "lowered to line up with the next checkpoint time: "
                  << m_output->getNextCheckpointTime();
        }

        if (invalidAll & CLAMP_TIME_TO_MAX) {
          if (!message.str().empty())
            message << std::endl;

          message << header.str()
                  << "for one or more ranks the next delta T was "
                  << "lowered to line up with the maximum simulation time of "
                  << m_simTimeMax;
        }
      }

      // Finally, output the summary.
      if (!message.str().empty()) {
        DOUT(true, message.str());
      }
    }
  }

  return invalid;
}

//______________________________________________________________________
//
// Determines if the time step is the last one. 
bool
ApplicationCommon::isLastTimeStep( double walltime )
{
  if (getReductionVariable(endSimulation_name)) {
    return true;
  }

  if (getReductionVariable(abortTimeStep_name)) {
    return true;
  }

  if (m_simTimeMax > 0 && m_simTime >= m_simTimeMax) {
    return true;
  }

  if (m_timeStepsMax > 0 && m_timeStep >= m_timeStepsMax) {
    return true;
  }

  if (m_wallTimeMax > 0) {
    // When using the wall clock time, rank 0 determines the time and
    // sends it to all other ranks.
    Uintah::MPI::Bcast(&walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm());

    if (walltime >= m_wallTimeMax) {
      return true;
    }
  }

  return false;
}

//______________________________________________________________________
//
// Determines if the time step may be the last one. The simulation
// time, d_delt, and the time step are known. The only real unknown is
// the wall time for the simulation calculation. The best guess is
// based on the ExpMovingAverage of the previous time steps.
//
// MaybeLast should be called before any time step work is done.

bool
ApplicationCommon::maybeLastTimeStep( double walltime ) const
{
  if (m_simTimeMax > 0 && m_simTime + m_delT >= m_simTimeMax) {
    return true;
  }

  if (m_timeStepsMax > 0 && m_timeStep + 1 >= m_timeStepsMax) {
    return true;
  }

  if (m_wallTimeMax > 0) {
    // When using the wall clock time, rank 0 determines the time and
    // sends it to all other ranks.
    Uintah::MPI::Bcast(&walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm());

    if (walltime >= m_wallTimeMax) {
      return true;
    }
  }

  return false;
}

//______________________________________________________________________
//
// This method is called only at restart or initialization -
// see SimulationController::timeStateSetup().

void ApplicationCommon::setTimeStep( int timeStep )
{
  m_timeStep = timeStep;

  // Write the time step to the initial DW so apps can get to it when scheduling.
  m_scheduler->getLastDW()->override(timeStep_vartype(m_timeStep), m_timeStepLabel );
}

//______________________________________________________________________
//
void ApplicationCommon::incrementTimeStep()
{
  ++m_timeStep;

  // Write the new time to the new data warehouse as the scheduler has
  // not yet advanced to the next data warehouse - see
  // SchedulerCommon::advanceDataWarehouse()
  m_scheduler->getLastDW()->override(timeStep_vartype(m_timeStep), m_timeStepLabel );
}

//______________________________________________________________________
//
// This method is called only at restart or initialization -
// see SimulationController::timeStateSetup().

void ApplicationCommon::setSimTime( double simTime )
{
  m_simTime = simTime;

  // Write the time step to the initial DW so apps can get to it when scheduling.
  m_scheduler->getLastDW()->override(simTime_vartype(m_simTime), m_simulationTimeLabel );
}

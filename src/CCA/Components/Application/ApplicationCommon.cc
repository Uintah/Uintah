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

ApplicationCommon::ApplicationCommon( const ProcessorGroup   * myworld,
                                      const SimulationStateP   sharedState )
    : UintahParallelComponent(myworld), m_sharedState(sharedState)
{
  // There should only be one SimulationState. If there is a single
  // application the ComponentFactory will pass in a null pointer
  // which will trigger the SimulationState to be created.

  // If there are multiple applications the Switcher (which is an
  // application) will create the SimulationState and then pass that
  // to the child applications.

  // If there are combined applications (aka MPMICE) it will create
  // the SimulationState and then pass that to the child applications.

  if (m_sharedState == nullptr) {
    m_sharedState = scinew SimulationState();
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

  // output interval
  VarLabel* nonconstOutputInv = VarLabel::create(outputInterval_name, min_vartype::getTypeDescription());
  nonconstOutputInv->allowMultipleComputes();
  m_outputIntervalLabel = nonconstOutputInv;

  // output time step interval
  VarLabel* nonconstOutputTimeStepInv = VarLabel::create(outputTimeStepInterval_name, min_vartype::getTypeDescription());
  nonconstOutputTimeStepInv->allowMultipleComputes();
  m_outputTimeStepIntervalLabel = nonconstOutputTimeStepInv;

  // check point interval
  VarLabel* nonconstCheckpointInv = VarLabel::create(checkpointInterval_name, min_vartype::getTypeDescription());
  nonconstCheckpointInv->allowMultipleComputes();
  m_checkpointIntervalLabel = nonconstCheckpointInv;

  // check point time step interval
  VarLabel* nonconstCheckpointTimeStepInv = VarLabel::create(checkpointTimeStepInterval_name, min_vartype::getTypeDescription());
  nonconstCheckpointTimeStepInv->allowMultipleComputes();
  m_checkpointTimeStepIntervalLabel = nonconstCheckpointTimeStepInv;

  // End Simulation  
  VarLabel* nonconstEndSimulation = VarLabel::create(endSimulation_name, bool_or_vartype::getTypeDescription());
  nonconstEndSimulation->allowMultipleComputes();
  m_endSimulationLabel = nonconstEndSimulation;

  m_application_stats.insert( CarcassCount, std::string("CarcassCount"), "Carcasses", 0 );
}

ApplicationCommon::~ApplicationCommon()
{
  releaseComponents();
  
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
  VarLabel::destroy(m_delTLabel);

  VarLabel::destroy(m_outputIntervalLabel);
  VarLabel::destroy(m_outputTimeStepIntervalLabel);
  VarLabel::destroy(m_checkpointIntervalLabel);
  VarLabel::destroy(m_checkpointTimeStepIntervalLabel);

  VarLabel::destroy(m_endSimulationLabel);
  
  if( m_simulationTime )
    delete m_simulationTime;

  // No need to delete the shared state as it is refcounted
  m_sharedState = nullptr;
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
  m_simulationTime = scinew SimulationTime(prob_spec);

  m_simTime = m_simulationTime->m_init_time;

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

  // An application may adjust the output interval or the checkpoint
  // interval during a simulation.  For example in deflagration ->
  // detonation simulations
  if (m_adjustOutputInterval) {
    task->requires(Task::NewDW, m_outputIntervalLabel);
  }

  if (m_adjustCheckpointInterval) {
    task->requires(Task::NewDW, m_checkpointIntervalLabel);
  }

  // An application may request that the simulation end early.
  if (m_mayEndSimulation) {
    task->requires(Task::NewDW, m_endSimulationLabel);
  }

  // The above three tasks are on a per proc basis any rank can make
  // the request because it is a either benign or a set value.
  scheduler->addTask(task, perProcPatchSet, m_sharedState->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::reduceSystemVars( const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * /*matls*/,
                                           DataWarehouse  * /*old_dw*/,
                                           DataWarehouse  * new_dw )
{
  // The goal of this task is to line up the delT across all levels.
  // If the coarse delT already exists (the one without an associated
  // level), then the application is not doing AMR.
  Patch* patch = nullptr;

  if (patches->size() != 0 && !new_dw->exists(m_delTLabel, -1, patch)) {
    int multiplier = 1;
    const GridP grid = patches->get(0)->getLevel()->getGrid();

    for (int i = 0; i < grid->numLevels(); i++) {
      const LevelP level = grid->getLevel(i);

      if (i > 0 && !m_lockstepAMR) {
        multiplier *= level->getRefinementRatioMaxDim();
      }

      if (new_dw->exists(m_delTLabel, -1, *level->patchesBegin())) {
        delt_vartype delTvar;
        new_dw->get(delTvar, m_delTLabel, level.get_rep());

        new_dw->put(delt_vartype(delTvar * multiplier), m_delTLabel);
      }
    }
  }

  if (d_myworld->nRanks() > 1) {
    new_dw->reduceMPI(m_delTLabel, 0, 0, -1);
  }

  // Validate the next delta T
  validateNextDelT( new_dw );

  // Reduce the output interval, checkpoint interval, and end
  // simulation variables. If no value was computed on an MPI rank, a
  // benign value will be set. If the reduction result is also a
  // benign value, that means no MPI rank wants to change the interval
  // and the value will be ignored.


  // An application may adjust the output interval or the checkpoint
  // interval during a simulation.  For example in deflagration ->
  // detonation simulations
  if (m_adjustOutputInterval) {
    if (patches->size() != 0 && !new_dw->exists(m_outputIntervalLabel, -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_outputIntervalLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_outputIntervalLabel, 0, 0, -1);
    }

    min_vartype outputInv_var;
    new_dw->get( outputInv_var, m_outputIntervalLabel );
      
    if( !outputInv_var.isBenignValue() ) {
      m_output->setOutputInterval( outputInv_var );
    }
  }

  if (m_adjustCheckpointInterval) {
    if (patches->size() != 0 && !new_dw->exists(m_checkpointIntervalLabel, -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_checkpointIntervalLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_checkpointIntervalLabel, 0, 0, -1);
    }

    min_vartype checkInv_var;
    new_dw->get( checkInv_var, m_checkpointIntervalLabel );
    
    if ( !checkInv_var.isBenignValue() ) {
      m_output->setCheckpointInterval( checkInv_var );
    }
  }

  // An application may request that the simulation end early.
  if (m_mayEndSimulation) {
    if (patches->size() != 0 && !new_dw->exists(m_endSimulationLabel, -1, patch)) {
      bool_or_vartype endSim;
      endSim.setBenignValue();
      new_dw->put(endSim, m_endSimulationLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_endSimulationLabel, 0, 0, -1);
    }

    bool_or_vartype endSim_var;
    new_dw->get( endSim_var, m_endSimulationLabel );
    
    if ( !endSim_var.isBenignValue() ) {
      m_endSimulation = endSim_var;
    }
  }
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

  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(),
                                      false, false, false, true, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(),
                                      false, false, false, true, true);
  // treatAsOld copyData noScrub notCopyData noCheckpoint

  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << grid->numLevels() << "  " 
  //        << scheduler->get_dw(0) << "  " 
  //        << scheduler->get_dw(1) << "  " 
  //        << scheduler->getLastDW() << std::endl;
  
  scheduler->addTask(task, perProcPatchSet, m_sharedState->allMaterials());
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
  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << new_dw << std::endl;  

  // Initialize the time step.
  new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

  // m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );  

  // Initialize the simulation time.
  new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);

  // m_sharedState->setElapsedSimTime( m_simTime );

  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << new_dw << std::endl;  
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

  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(),
                                      false, false, false, true, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(),
                                      false, false, false, true, true);
  // treatAsOld copyData noScrub notCopyData noCheckpoint

  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << grid->numLevels() << "  " 
  //        << scheduler->get_dw(0) << "  " 
  //        << scheduler->get_dw(1) << "  " 
  //        << scheduler->getLastDW() << std::endl;
    
  scheduler->addTask(task, perProcPatchSet, m_sharedState->allMaterials());
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
  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << new_dw << std::endl;  

  // If the time step is being restarted do not update the simulation
  // time. The time step does not get up dated here but is stored so
  // it can be at the top of the simcontroller loop.
  
  if (!new_dw->timestepRestarted())
  {
    // Store the time step so it can be incremented at the top of the
    // time step where it is over written.
    new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

    // m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );  
    
    // // Update the simulation time.
    m_simTime += m_delT;
    new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);

    // m_sharedState->setElapsedSimTime( m_simTime );  
  }

  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << new_dw << std::endl;  
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
int
ApplicationCommon::computeTaskGraphIndex()
{
  // Call the actual application method which if defined overrides the
  // virtual default method for the index
  return computeTaskGraphIndex( m_timeStep );
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

  proc0cout << "Restarting time step at " << m_simTime
            << ", changing delT from " << m_delT
            << " to " << new_delT
            << std::endl;

  // Bulletproofing
  if (new_delT < m_simulationTime->m_delt_min || new_delT <= 0) {
    std::ostringstream warn;
    warn << "The new delT (" << new_delT << ") is either less than "
         << "delT_min (" << m_simulationTime->m_delt_min << ") or equal to 0";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  
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
bool
ApplicationCommon::needRecompile( const GridP& /*grid*/)
{
#ifdef HAVE_VISIT
  // Check all of the application variables that might require the task
  // graph to be recompiled.
  for( unsigned int i=0; i<getUPSVars().size(); ++i )
  {
    ApplicationInterface::interactiveVar &var = getUPSVars()[i];
    
    if( var.modified && var.recompile )
    {
      m_recompile = true;
      break;
    }
  }
#endif
  
  if( m_recompile ) {
    m_recompile = false;
    return true;
  }
  else {
    return false;
  }
}

//______________________________________________________________________
//
void
ApplicationCommon::prepareForNextTimeStep()
{
  // Increment (by one) the current time step number so components
  // know what time step they are on and get the delta T that will be
  // used.
  incrementTimeStep();

  // Get the delta that will be used for the time step.
  delt_vartype delt_var;
  m_scheduler->getLastDW()->get( delt_var, m_delTLabel );
  m_delT = delt_var;
}

//______________________________________________________________________
//
void
ApplicationCommon::setDelTForAllLevels( SchedulerP& scheduler,
                                        const GridP & grid,
                                        const int totalFine )
{
  // Adjust the delT for each level and store it in all applicable dws.
  double delT_fine = m_delT;
  int skip         = totalFine;
  
  for (int i = 0; i < grid->numLevels(); ++i)
  {
    const Level* level = grid->getLevel(i).get_rep();
    
    if( isAMR() && i != 0 && !isLockstepAMR() )
    {
      int trr = level->getRefinementRatioMaxDim();
      delT_fine /= trr;
      skip /= trr;
    }
    
    for (int idw = 0; idw < totalFine; idw += skip)
    {
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
// This method is called only at restart -
// see SimulationController::timeStateSetup().

void
ApplicationCommon::setNextDelT( double delT )
{
  // Restart - Check to see if the user has set a restart delT.
  if (m_simulationTime->m_override_restart_delt != 0) {
    proc0cout << "Overriding restart delT " << m_delT << " with "
              << m_simulationTime->m_override_restart_delt << "\n";
    
    m_nextDelT = m_simulationTime->m_override_restart_delt;

    m_scheduler->getLastDW()->override(delt_vartype(m_nextDelT), m_delTLabel);
  }

  // Restart - Otherwise get the next delta T from the archive.
  else if( m_scheduler->getLastDW()->exists( m_delTLabel ) )
  {
    delt_vartype delt_var;
    m_scheduler->getLastDW()->get( delt_var, m_delTLabel );
    m_nextDelT = delt_var;
  }

  // Restart - All else fails use the previous delta T.
  else
  {
    m_nextDelT = delT;
    m_scheduler->getLastDW()->override(delt_vartype(m_nextDelT), m_delTLabel);
  }

  // std::cerr << "*************" << __FUNCTION__ << "  " << __LINE__ << "  " << m_nextDelT << "  " << delT << std::endl;
}

//______________________________________________________________________
//
void
ApplicationCommon::validateNextDelT( DataWarehouse* newDW )
{
  // Retrieve the proposed next delta T and adjust it based on
  // simulation time info parameters.

  // NOTE: This check is performed BEFORE the simulation time is
  // updated. As such, being that the time step has completed, the
  // actual simulation time is the current simulation time plus the
  // current delta T.

  delt_vartype delt_var;
  newDW->get( delt_var, m_delTLabel );
  m_nextDelT = delt_var;

  // Adjust the next delT by the factor
  m_nextDelT *= m_simulationTime->m_delt_factor;
      
  // Check to see if the next delT is below the delt_min
  if( m_nextDelT < m_simulationTime->m_delt_min ) {
    proc0cout << "WARNING: raising delT from " << m_nextDelT;
    
    m_nextDelT = m_simulationTime->m_delt_min;
    
    proc0cout << " to minimum: " << m_nextDelT << '\n';
  }

  // Check to see if the next delT was increased too much over the
  // current delT
  double delt_tmp = (1.0+m_simulationTime->m_max_delt_increase) * m_delT;
  
  if( m_delT > 0.0 &&
      m_simulationTime->m_max_delt_increase > 0 &&
      m_nextDelT > delt_tmp ) {
    proc0cout << "WARNING (a): lowering delT from " << m_nextDelT;
    
    m_nextDelT = delt_tmp;
    
    proc0cout << " to maxmimum: " << m_nextDelT
              << " (maximum increase of " << m_simulationTime->m_max_delt_increase
              << ")\n";
  }

  // Check to see if the next delT exceeds the max_initial_delt
  if( m_simulationTime->m_max_initial_delt > 0 &&
      m_simTime+m_delT <= m_simulationTime->m_initial_delt_range &&
      m_nextDelT > m_simulationTime->m_max_initial_delt ) {
    proc0cout << "WARNING (b): lowering delT from " << m_nextDelT ;

    m_nextDelT = m_simulationTime->m_max_initial_delt;

    proc0cout<< " to maximum: " << m_nextDelT
             << " (for initial timesteps)\n";
  }

  // Check to see if the next delT exceeds the delt_max
  if( m_nextDelT > m_simulationTime->m_delt_max ) {
    proc0cout << "WARNING (c): lowering delT from " << m_nextDelT;

    m_nextDelT = m_simulationTime->m_delt_max;
    
    proc0cout << " to maximum: " << m_nextDelT << '\n';
  }

  // Clamp the next delT to match the requested output and/or
  // checkpoint times
  if( m_simulationTime->m_clamp_time_to_output ) {

    // Clamp to the output time
    double nextOutput = m_output->getNextOutputTime();
    if (nextOutput != 0 && m_simTime + m_delT + m_nextDelT > nextOutput) {
      proc0cout << "WARNING (d): lowering delT from " << m_nextDelT;

      m_nextDelT = nextOutput - (m_simTime+m_delT);

      proc0cout << " to " << m_nextDelT
                << " to line up with output time\n";
    }

    // Clamp to the checkpoint time
    double nextCheckpoint = m_output->getNextCheckpointTime();
    if (nextCheckpoint != 0 && m_simTime + m_delT + m_nextDelT > nextCheckpoint) {
      proc0cout << "WARNING (d): lowering delT from " << m_nextDelT;

      m_nextDelT = nextCheckpoint - (m_simTime+m_delT);

      proc0cout << " to " << m_nextDelT
                << " to line up with checkpoint time\n";
    }
  }
  
  // Clamp delT to the max end time,
  if (m_simulationTime->m_end_at_max_time &&
      m_simTime + m_delT + m_nextDelT > m_simulationTime->m_max_time) {
    m_nextDelT = m_simulationTime->m_max_time - (m_simTime+m_delT);
  }

  // Write the next delT to the data warehouse
  newDW->override( delt_vartype(m_nextDelT), m_delTLabel );
}

//______________________________________________________________________
//
// Determines if the time step is the last one. 
bool
ApplicationCommon::isLastTimeStep( double walltime ) const
{
  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

  return ( m_endSimulation ||

           ( m_simTime >= m_simulationTime->m_max_time ) ||

           ( m_simulationTime->m_max_time_steps > 0 &&
             m_timeStep >= m_simulationTime->m_max_time_steps ) ||

           ( m_simulationTime->m_max_wall_time > 0 &&
             walltime >= m_simulationTime->m_max_wall_time ) );
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
  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );
  
  return ( (m_simTime + m_delT >= m_simulationTime->m_max_time) ||
           
           (m_simulationTime->m_max_time_steps > 0 &&
            m_timeStep + 1 >= m_simulationTime->m_max_time_steps) ||
           
           (m_simulationTime->m_max_wall_time > 0 &&
            walltime >= m_simulationTime->m_max_wall_time) );
}

//______________________________________________________________________
//
// This method is called only at restart or initialization -
// see SimulationController::timeStateSetup().

void ApplicationCommon::setTimeStep( int timeStep )
{
  m_timeStep = timeStep;

  // Write the time step to the inital DW so apps can get to it when
  // scheduling.
  m_scheduler->getLastDW()->override(timeStep_vartype(m_timeStep),
                                     m_timeStepLabel );
  
  // m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );
}

//______________________________________________________________________
//
void ApplicationCommon::incrementTimeStep()
{
  ++m_timeStep;

  // Write the new time to the new data warehouse as the scheduler has
  // not yet advanced to the next data warehouse - see
  // SchedulerCommon::advanceDataWarehouse()
  DataWarehouse* newDW = m_scheduler->getLastDW();

  newDW->override(timeStep_vartype(m_timeStep), m_timeStepLabel );

  // m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );
}

//______________________________________________________________________
//
// This method is called only at restart or initialization -
// see SimulationController::timeStateSetup().

void ApplicationCommon::setSimTime( double simTime )
{
  m_simTime = simTime;

  // Write the time step to the inital DW so apps can get to it when
  // scheduling.
  m_scheduler->getLastDW()->override(simTime_vartype(m_simTime),
                                     m_simulationTimeLabel );
  
  // m_sharedState->setElapsedSimTime( m_simTime );
}

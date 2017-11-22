
/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

ApplicationCommon::ApplicationCommon(const ProcessorGroup* myworld,
				     const SimulationStateP sharedState) :
  UintahParallelComponent(myworld), m_sharedState(sharedState)
{
  // There should only be one SimulationState. If there is a single
  // application the ComponentFactory will pass in a null pointer
  // which will trigger the SimulationState to be created.

  // If there are multiple applications the Switcher (which is an
  // application) will create the SimulationState and then pass that
  // to the other applications.

  // If there are combined applications (aka MPMICE) it will create
  // the SimulationState and then pass that to the other applications.
  
  if( m_sharedState == nullptr )
    m_sharedState = scinew SimulationState();
  
  //__________________________________
  //  These variables can be modified by an application.

  // Time Step
  m_timeStepLabel =
    VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );

  // Simulation Time
  m_simulationTimeLabel =
    VarLabel::create(simTime_name, simTime_vartype::getTypeDescription() );

  // delta t
  VarLabel* nonconstDelT =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
  nonconstDelT->allowMultipleComputes();
  m_delTLabel = nonconstDelT;

  // output interval
  VarLabel* nonconstOutputInv =
    VarLabel::create(outputInterval_name,
		     min_vartype::getTypeDescription() );
  nonconstOutputInv->allowMultipleComputes();
  m_outputIntervalLabel = nonconstOutputInv;

  // output time step interval
  VarLabel* nonconstOutputTimeStepInv =
    VarLabel::create(outputTimeStepInterval_name,
  		     min_vartype::getTypeDescription() );
  nonconstOutputTimeStepInv->allowMultipleComputes();
  m_outputTimeStepIntervalLabel = nonconstOutputTimeStepInv;

  // check point interval
  VarLabel* nonconstCheckpointInv =
    VarLabel::create(checkpointInterval_name,
		     min_vartype::getTypeDescription() );
  nonconstCheckpointInv->allowMultipleComputes();
  m_checkpointIntervalLabel = nonconstCheckpointInv;
  
  // check point time step interval
  VarLabel* nonconstCheckpointTimeStepInv =
    VarLabel::create(checkpointTimeStepInterval_name,
  		     min_vartype::getTypeDescription() );
  nonconstCheckpointTimeStepInv->allowMultipleComputes();
  m_checkpointTimeStepIntervalLabel = nonconstCheckpointTimeStepInv;

  // End Simulation  
  VarLabel* nonconstEndSimulation =
    VarLabel::create(endSimulation_name,
		     bool_or_vartype::getTypeDescription() );
  nonconstEndSimulation->allowMultipleComputes();
  m_endSimulationLabel = nonconstEndSimulation;
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

  if( m_sharedState.get_rep() )
    delete m_sharedState.get_rep();
}

void ApplicationCommon::setComponents( const ApplicationCommon *parent,
				       const ProblemSpecP &prob_spec )
{
  attachPort( "scheduler", parent->m_scheduler );
  attachPort( "solver",    parent->m_solver );
  attachPort( "regridder", parent->m_regridder );
  attachPort( "output",    parent->m_output );

  getComponents();

  problemSetup( prob_spec );
}

void ApplicationCommon::getComponents()
{
  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( isDynamicRegridding() && !m_scheduler ) {
    throw InternalError("dynamic_cast of 'd_regridder' failed!",
                        __FILE__, __LINE__);
  }

  m_solver = dynamic_cast<SolverInterface*>( getPort("solver") );

  if( !m_solver ) {
    throw InternalError("dynamic_cast of 'd_solver' failed!",
                        __FILE__, __LINE__);
  }

  m_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( isDynamicRegridding() && !m_regridder ) {
    throw InternalError("dynamic_cast of 'd_regridder' failed!",
                        __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!",
                        __FILE__, __LINE__);
  }
}

void ApplicationCommon::releaseComponents()
{
  m_scheduler = nullptr;
  m_solver    = nullptr;
  m_regridder = nullptr;
  m_output    = nullptr;
}

void ApplicationCommon::problemSetup( const ProblemSpecP &prob_spec )
{
  m_simulationTime = scinew SimulationTime( prob_spec );

  m_simTime = m_simulationTime->m_init_time;

  // Check for an AMR attribute with the grid.
  ProblemSpecP grid_ps = prob_spec->findBlock( "Grid" );

  if( grid_ps ) {
    grid_ps->getAttribute( "doAMR", m_AMR );

    m_dynamicRegridding = m_AMR;
  }

  // If the AMR block is defined default to turning AMR on.
  ProblemSpecP amr_ps = prob_spec->findBlock( "AMR" );
  
  if( amr_ps ) {
    m_AMR = true;

    std::string type;
    amr_ps->getAttribute( "type", type );

    m_dynamicRegridding = (type.empty() || type == std::string( "Dynamic" ));

    amr_ps->get( "useLockStep", m_lockstepAMR );
  }
}

void
ApplicationCommon::scheduleRefine(const PatchSet*,
				  SchedulerP&)
{
  throw InternalError( "scheduleRefine not implemented for this application\n", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleRefineInterface(const LevelP&,
					   SchedulerP&,
					   bool, bool)
{
  throw InternalError( "scheduleRefineInterface is not implemented for this application\n", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleCoarsen(const LevelP&,
				   SchedulerP&)
{
  throw InternalError( "scheduleCoarsen is not implemented for this application\n", __FILE__, __LINE__ );
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
  MALLOC_TRACE_TAG_SCOPE("ApplicationCommon::reduceSysVar()");

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
        double delT;
        new_dw->get(delTvar, m_delTLabel, level.get_rep());

        new_dw->put(delt_vartype(delTvar * multiplier), m_delTLabel);
      }
    }
  }

  if (d_myworld->nRanks() > 1) {
    new_dw->reduceMPI(m_delTLabel, 0, 0, -1);
  }

  // Reduce the output interval, checkpoint interval, and end
  // simulation variables. If no value was computed on an MPI rank, a
  // benign value will be set. If the reduction result is also a
  // benign value, that means no MPI rank wants to change the interval
  // and the value will be ignored.


  // An application may adjust the output interval or the checkpoint
  // interval during a simulation.  For example in deflagration ->
  // detonation simulations
  if (m_adjustOutputInterval) {
    if (patches->size() != 0 &&
	!new_dw->exists(m_outputIntervalLabel, -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_outputIntervalLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_outputIntervalLabel, 0, 0, -1);
    }
  }

  if (m_adjustCheckpointInterval) {
    if (patches->size() != 0 &&
	!new_dw->exists(m_checkpointIntervalLabel, -1, patch)) {
      min_vartype inv;
      inv.setBenignValue();
      new_dw->put(inv, m_checkpointIntervalLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_checkpointIntervalLabel, 0, 0, -1);
    }
  }

  // An application may request that the simulation end early.
  if (m_mayEndSimulation) {
    if (patches->size() != 0 &&
	!new_dw->exists(m_endSimulationLabel, -1, patch)) {
      max_vartype endSim;
      endSim.setBenignValue();
      new_dw->put(endSim, m_endSimulationLabel);
    }

    if (d_myworld->nRanks() > 1) {
      new_dw->reduceMPI(m_endSimulationLabel, 0, 0, -1);
    }
  }
}  // end reduceSysVar()


//______________________________________________________________________
//
void
ApplicationCommon::finalizeSystemVars( SchedulerP& scheduler )
{
  // Get the next delta T - Do this before reporting stats or the
  // in-situ so the new delT is availble.
  getNextDelT( scheduler );
  
  // An application may update the output interval or the checkpoint
  // interval during a simulation.  For example in deflagration ->
  // detonation simulations
  DataWarehouse* newDW = scheduler->getLastDW();

  if( m_adjustOutputInterval ) {
    min_vartype outputInv_var;
    newDW->get( outputInv_var, m_outputIntervalLabel );
      
    if( !outputInv_var.isBenignValue() ) {
      m_output->setOutputInterval( outputInv_var );
    }
  }

  if( m_adjustCheckpointInterval ) {
    min_vartype checkInv_var;
    newDW->get( checkInv_var, m_checkpointIntervalLabel );
    
    if ( !checkInv_var.isBenignValue() ) {
      m_output->setCheckpointInterval( checkInv_var );
    }
  }

  // An application may request that the simulation end early.
  if( m_mayEndSimulation ) {
    max_vartype endSim_var;
    newDW->get( endSim_var, m_endSimulationLabel );
    
    if ( !endSim_var.isBenignValue() ) {
      m_endSimulation = endSim_var;
    }
  }
}


//______________________________________________________________________
//
void
ApplicationCommon::scheduleInitializeSystemVars(const GridP& grid,
						const PatchSet* perProcPatchSet,
						SchedulerP& scheduler)
{
  // Initialize the system vars which are on a per rank basis.
  Task* task = scinew Task("ApplicationCommon::initializeSystemVars", this,
			   &ApplicationCommon::initializeSystemVars);

  task->setType(Task::OncePerProc);
  task->computes(m_timeStepLabel);
  task->computes(m_simulationTimeLabel);

  // for (int i = 0; i < grid->numLevels(); i++) {
  //   task->computes(m_timeStepLabel,       grid->getLevel(i).get_rep());
  //   task->computes(m_simulationTimeLabel, grid->getLevel(i).get_rep());
  // }
  
  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(),
				      false, false, true, false, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(),
				      false, false, true, false, true);
  // treatAsOld copyData noScrub notCopyData noCheckpoint

  // std::cerr << __FUNCTION__ << "  "
  // 	    << grid->numLevels() << "  " 
  // 	    << scheduler->get_dw(0) << "  " 
  // 	    << scheduler->get_dw(1) << "  " 
  // 	    << scheduler->getLastDW() << std::endl;
    
  scheduler->addTask(task, perProcPatchSet, m_sharedState->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::initializeSystemVars( const ProcessorGroup *,
					 const PatchSubset    * patches,
					 const MaterialSubset * /*matls*/,
					       DataWarehouse  * old_dw,
				               DataWarehouse  * new_dw )
{
  MALLOC_TRACE_TAG_SCOPE("ApplicationCommon::initializeSystemVar()");

  // const Level* level = getLevel(patches);
  
  // Initialize the time step.
  new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

  m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );  

  // Initialize the simulation time.
  new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);

  // new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel, level);

  m_sharedState->setElapsedSimTime( m_simTime );  
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

  // for (int i = 0; i < grid->numLevels(); i++) {
  //   task->computes(m_timeStepLabel,       grid->getLevel(i).get_rep());
  //   task->computes(m_simulationTimeLabel, grid->getLevel(i).get_rep());
  // }

  scheduler->overrideVariableBehavior(m_timeStepLabel->getName(),
				      false, false, true, false, true);
  scheduler->overrideVariableBehavior(m_simulationTimeLabel->getName(),
				      false, false, true, false, true);
  // treatAsOld copyData noScrub notCopyData noCheckpoint

  // std::cerr << __FUNCTION__ << "  "
  // 	    << grid->numLevels() << "  " 
  // 	    << scheduler->get_dw(0) << "  " 
  // 	    << scheduler->get_dw(1) << "  " 
  // 	    << scheduler->getLastDW() << std::endl;
    
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
  MALLOC_TRACE_TAG_SCOPE("ApplicationCommon::updateSystemVar()");

  // If the time step is being restarted do not update the simulation
  // time. The time step does not get up dated here but is stored so
  // it can be at the top of teh simcontroller loop.
  
  if (!new_dw->timestepRestarted())
  {
    // const Level* level = getLevel(patches);
  
    // Store the time step so it can be incremented at the top of the
    // time step where it is over written.
    new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

    m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );  
    
    // Update the simulation time.
    m_simTime += m_delT;
    new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);
    // new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel, level);
    
    m_sharedState->setElapsedSimTime( m_simTime );  
  }
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
ApplicationCommon::recomputeTimeStep(double)
{
  throw InternalError("recomputeTimestep is not implemented for this application", __FILE__, __LINE__);
}

//______________________________________________________________________
//
bool
ApplicationCommon::restartableTimeSteps()
{
  return false;
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
ApplicationCommon::getNextDelT( SchedulerP& scheduler )
{
  m_prevDelT = m_delT;

  // Retrieve the next delta T and adjust it based on timeinfo
  // parameters.
  DataWarehouse* newDW = scheduler->getLastDW();
						   
  delt_vartype delt_var;
  newDW->get( delt_var, m_delTLabel );
  m_delT = delt_var;

  // Adjust the delt
  m_delT *= m_simulationTime->m_delt_factor;
      
  // Check to see if the new delT is below the delt_min
  if( m_delT < m_simulationTime->m_delt_min ) {
    proc0cout << "WARNING: raising delT from " << m_delT;
    
    m_delT = m_simulationTime->m_delt_min;
    
    proc0cout << " to minimum: " << m_delT << '\n';
  }

  // Check to see if the new delT was increased too much over the
  // previous delt
  double delt_tmp = (1.0+m_simulationTime->m_max_delt_increase) * m_prevDelT;
  
  if( m_prevDelT > 0.0 &&
      m_simulationTime->m_max_delt_increase > 0 &&
      m_delT > delt_tmp ) {
    proc0cout << "WARNING (a): lowering delT from " << m_delT;
    
    m_delT = delt_tmp;
    
    proc0cout << " to maxmimum: " << m_delT
              << " (maximum increase of " << m_simulationTime->m_max_delt_increase
              << ")\n";
  }

  // Check to see if the new delT exceeds the max_initial_delt
  if( m_simTime <= m_simulationTime->m_initial_delt_range &&
      m_simulationTime->m_max_initial_delt > 0 &&
      m_delT > m_simulationTime->m_max_initial_delt ) {
    proc0cout << "WARNING (b): lowering delT from " << m_delT ;

    m_delT = m_simulationTime->m_max_initial_delt;

    proc0cout<< " to maximum: " << m_delT
             << " (for initial timesteps)\n";
  }

  // Check to see if the new delT exceeds the delt_max
  if( m_delT > m_simulationTime->m_delt_max ) {
    proc0cout << "WARNING (c): lowering delT from " << m_delT;

    m_delT = m_simulationTime->m_delt_max;
    
    proc0cout << " to maximum: " << m_delT << '\n';
  }

  // Clamp delT to match the requested output and/or checkpoint times
  if( m_simulationTime->m_clamp_time_to_output ) {

    // Clamp to the output time
    double nextOutput = m_output->getNextOutputTime();
    if (nextOutput != 0 && m_simTime + m_delT > nextOutput) {
      proc0cout << "WARNING (d): lowering delT from " << m_delT;

      m_delT = nextOutput - m_simTime;

      proc0cout << " to " << m_delT
                << " to line up with output time\n";
    }

    // Clamp to the checkpoint time
    double nextCheckpoint = m_output->getNextCheckpointTime();
    if (nextCheckpoint != 0 && m_simTime + m_delT > nextCheckpoint) {
      proc0cout << "WARNING (d): lowering delT from " << m_delT;

      m_delT = nextCheckpoint - m_simTime;

      proc0cout << " to " << m_delT
                << " to line up with checkpoint time\n";
    }
  }
  
  // Clamp delT to the max end time,
  if (m_simulationTime->m_end_at_max_time &&
      m_simTime + m_delT > m_simulationTime->m_max_time) {
    m_delT = m_simulationTime->m_max_time - m_simTime;
  }

  // Write the new delT to the data warehouse
  newDW->override( delt_vartype(m_delT), m_delTLabel );
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
void ApplicationCommon::setTimeStep( int timeStep )
{
  m_timeStep = timeStep;
  m_sharedState->setCurrentTopLevelTimeStep( timeStep );
}

//______________________________________________________________________
//
void ApplicationCommon::incrementTimeStep()
{
  ++m_timeStep;
  m_sharedState->setCurrentTopLevelTimeStep( m_timeStep );

  // Write the new time to the data warehouse
  DataWarehouse* newDW = m_scheduler->getLastDW();
  newDW->override(timeStep_vartype(m_timeStep), m_timeStepLabel);
}

//______________________________________________________________________
//
void ApplicationCommon::setSimTime( double val )
{
  m_simTime = val;
  m_sharedState->setElapsedSimTime( m_simTime );
}

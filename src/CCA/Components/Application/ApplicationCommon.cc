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

#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

using namespace Uintah;

ApplicationCommon::ApplicationCommon(const ProcessorGroup* myworld,
				     const SimulationStateP sharedState) :
  UintahParallelComponent(myworld), m_sharedState(sharedState)
{
  if( m_sharedState == nullptr )
    m_sharedState = scinew SimulationState();

  m_AMR                       = false;
  m_lockstepAMR               = false;
  
  m_dynamicRegridding         = false;
  
  m_haveModifiedVars          = false;
  
  m_updateCheckpointInterval  = false;
  m_updateOutputInterval      = false;

  m_simulationTime            = nullptr;
  
  VarLabel* nonconstDelt = 
    VarLabel::create("delT", delt_vartype::getTypeDescription() );

  nonconstDelt->allowMultipleComputes();
  m_delt_label = nonconstDelt;

  //__________________________________
  //  These variables can be modified by a component.
  VarLabel* nonconstOutputInv =             // output interval
    VarLabel::create("outputInterval",
		     min_vartype::getTypeDescription() );
  VarLabel* nonconstOutputTimestepInv =     // output timestep interval
    VarLabel::create("outputTimestepInterval",
  		     min_vartype::getTypeDescription() );

  VarLabel* nonconstCheckpointInv =         // check point interval
    VarLabel::create("checkpointInterval",
		     min_vartype::getTypeDescription() );
  
  VarLabel* nonconstCheckpointTimestepInv = // check point timestep interval
    VarLabel::create("checkpointTimestepInterval",
  		     min_vartype::getTypeDescription() );

  nonconstOutputInv->allowMultipleComputes();
  nonconstOutputTimestepInv->allowMultipleComputes();

  nonconstCheckpointInv->allowMultipleComputes();
  nonconstCheckpointTimestepInv->allowMultipleComputes();

  m_outputIntervalLabel             = nonconstOutputInv;
  m_outputTimestepIntervalLabel     = nonconstOutputTimestepInv;

  m_checkpointIntervalLabel         = nonconstCheckpointInv;
  m_checkpointTimestepIntervalLabel = nonconstCheckpointTimestepInv;

#ifdef HAVE_VISIT
  m_doVisIt = false;
#endif
}

ApplicationCommon::~ApplicationCommon()
{
  VarLabel::destroy(m_delt_label);

  VarLabel::destroy(m_outputIntervalLabel);
  VarLabel::destroy(m_outputTimestepIntervalLabel);
  VarLabel::destroy(m_checkpointIntervalLabel);
  VarLabel::destroy(m_checkpointTimestepIntervalLabel);

  if( m_simulationTime )
    delete m_simulationTime;
}

void ApplicationCommon::getComponents()
{
  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!",
                        __FILE__, __LINE__);
  }

  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!",
                        __FILE__, __LINE__);
  }
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
  throw InternalError( "scheduleRefine not implemented for this component\n", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleRefineInterface(const LevelP&, 
                                             SchedulerP&,
                                             bool, bool)
{
  throw InternalError( "scheduleRefineInterface is not implemented for this component\n", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleCoarsen(const LevelP&, 
                                     SchedulerP&)
{
  throw InternalError( "scheduleCoarsen is not implemented for this component\n", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleTimeAdvance(const LevelP&,
                                         SchedulerP&)
{
  throw InternalError( "scheduleTimeAdvance is not implemented for this component", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleErrorEstimate( const LevelP&,
                                                  SchedulerP& )
{
  throw InternalError( "scheduleErrorEstimate is not implemented for this component", __FILE__, __LINE__ );
}

void
ApplicationCommon::scheduleInitialErrorEstimate(const LevelP& /*coarseLevel*/,
                                                  SchedulerP& /*sched*/)
{
  throw InternalError("scheduleInitialErrorEstimate is not implemented for this component", __FILE__, __LINE__);
}

double
ApplicationCommon::recomputeTimestep(double)
{
  throw InternalError("recomputeTimestep is not implemented for this component", __FILE__, __LINE__);
}

bool
ApplicationCommon::restartableTimesteps()
{
  return false;
}

double
ApplicationCommon::getSubCycleProgress(DataWarehouse* fineDW)
{
  // DWs are always created in order of time.
  int fineID = fineDW->getID();  
  int coarseNewID = fineDW->getOtherDataWarehouse(Task::CoarseNewDW)->getID();
  // need to do this check, on init timestep, old DW is nullptr, and getOtherDW will throw exception
  if (fineID == coarseNewID) {
    return 1.0;
  }
  int coarseOldID = fineDW->getOtherDataWarehouse(Task::CoarseOldDW)->getID();
  
  return ((double)fineID-coarseOldID) / (coarseNewID-coarseOldID);
}

//______________________________________________________________________
//
void
ApplicationCommon::updateSimTime( void )
{
  m_simTime += m_delt;
}

//______________________________________________________________________
//
void
ApplicationCommon::getNextDeltaT( void )
{
  m_prev_delt = m_delt;

  // Retrieve the next delta T and adjust it based on timeinfo
  // parameters.
  DataWarehouse* newDW = m_scheduler->getLastDW();
						   
  delt_vartype delt_var;
  newDW->get( delt_var, m_delt_label );
  m_delt = delt_var;

  // Adjust the delt
  m_delt *= m_simulationTime->m_delt_factor;
      
  // Check to see if the new delt is below the delt_min
  if( m_delt < m_simulationTime->m_delt_min ) {
    proc0cout << "WARNING: raising delt from " << m_delt;
    
    m_delt = m_simulationTime->m_delt_min;
    
    proc0cout << " to minimum: " << m_delt << '\n';
  }

  // Check to see if the new delt was increased too much over the
  // previous delt
  double delt_tmp = (1.0+m_simulationTime->m_max_delt_increase) * m_prev_delt;
  
  if( m_prev_delt > 0.0 &&
      m_simulationTime->m_max_delt_increase > 0 &&
      m_delt > delt_tmp ) {
    proc0cout << "WARNING (a): lowering delt from " << m_delt;
    
    m_delt = delt_tmp;
    
    proc0cout << " to maxmimum: " << m_delt
              << " (maximum increase of " << m_simulationTime->m_max_delt_increase
              << ")\n";
  }

  // Check to see if the new delt exceeds the max_initial_delt
  if( m_simTime <= m_simulationTime->m_initial_delt_range &&
      m_simulationTime->m_max_initial_delt > 0 &&
      m_delt > m_simulationTime->m_max_initial_delt ) {
    proc0cout << "WARNING (b): lowering delt from " << m_delt ;

    m_delt = m_simulationTime->m_max_initial_delt;

    proc0cout<< " to maximum: " << m_delt
             << " (for initial timesteps)\n";
  }

  // Check to see if the new delt exceeds the delt_max
  if( m_delt > m_simulationTime->m_delt_max ) {
    proc0cout << "WARNING (c): lowering delt from " << m_delt;

    m_delt = m_simulationTime->m_delt_max;
    
    proc0cout << " to maximum: " << m_delt << '\n';
  }

  // Clamp delt to match the requested output and/or checkpoint times
  if( m_simulationTime->m_clamp_time_to_output ) {

    // Clamp to the output time
    double nextOutput = m_output->getNextOutputTime();
    if (nextOutput != 0 && m_simTime + m_delt > nextOutput) {
      proc0cout << "WARNING (d): lowering delt from " << m_delt;

      m_delt = nextOutput - m_simTime;

      proc0cout << " to " << m_delt
                << " to line up with output time\n";
    }

    // Clamp to the checkpoint time
    double nextCheckpoint = m_output->getNextCheckpointTime();
    if (nextCheckpoint != 0 && m_simTime + m_delt > nextCheckpoint) {
      proc0cout << "WARNING (d): lowering delt from " << m_delt;

      m_delt = nextCheckpoint - m_simTime;

      proc0cout << " to " << m_delt
                << " to line up with checkpoint time\n";
    }
  }
  
  // Clamp delt to the max end time,
  if (m_simulationTime->m_end_at_max_time &&
      m_simTime + m_delt > m_simulationTime->m_max_time) {
    m_delt = m_simulationTime->m_max_time - m_simTime;
  }

  // Write the new delt to the data warehouse
  newDW->override( delt_vartype(m_delt), m_delt_label );
}

//______________________________________________________________________
//
// Determines if the time step was the last one. 
bool
ApplicationCommon::isLast( double walltime, int currentTimeStep ) const
{
  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

  return ( ( m_simTime >= m_simulationTime->m_max_time ) ||

           ( m_simulationTime->m_max_timestep > 0 &&
             currentTimeStep >= m_simulationTime->m_max_timestep ) ||

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
ApplicationCommon::maybeLast( double walltime, int currentTimeStep ) const
{  
  // When using the wall clock time, rank 0 determines the time and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );
  
  return ( (m_simTime + m_delt >= m_simulationTime->m_max_time) ||
	   
	   (currentTimeStep + 1 >= m_simulationTime->m_max_timestep) ||
	   
	   (m_simulationTime->m_max_wall_time > 0 &&
	    walltime >= m_simulationTime->m_max_wall_time) );
}


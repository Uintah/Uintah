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

Uintah::Dout g_deltaT_minor_warnings( "DeltaTMinorWarnings", "ApplicationCommon", "Report minor warnings when validating the next delta T", true );

Uintah::Dout g_deltaT_major_warnings( "DeltaTMajorWarnings", "ApplicationCommon", "Report major warnings when validating the next delta T", true );

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
 
  m_application_stats.insert( CarcassCount, std::string("CarcassCount"), "Carcasses", 0 );
}

ApplicationCommon::~ApplicationCommon()
{
  releaseComponents();
  
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
  VarLabel::destroy(m_delTLabel);

  if( m_simulationTime )
    delete m_simulationTime;

  for ( auto & var : m_appReductionVars )
    delete var.second;
  
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

void ApplicationCommon::setReductionVariables( UintahParallelComponent *comp )
{
  ApplicationCommon * child = dynamic_cast<ApplicationCommon*>( comp );

  // Get the reduction active flags from the child;
  for ( auto & var : m_appReductionVars )
    var.second->active = child->m_appReductionVars[ var.first ]->active;
}

void ApplicationCommon::clearReductionVariables()
{
  for ( auto & var : m_appReductionVars )
    var.second->setBenignValue();
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

  // An application may also request that the time step be recomputed
  // or aborted or the simulation end early.
  for ( auto & var : m_appReductionVars )
  {
    if( var.second->active )
      task->requires(Task::NewDW, var.second->label);
  }

  // The above three tasks are on a per proc basis any rank can make
  // the request because it is a either benign or a set value.
  scheduler->addTask(task, perProcPatchSet, m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void
ApplicationCommon::reduceSystemVars( const ProcessorGroup *,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * /*matls*/,
                                           DataWarehouse  * old_dw,
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

  // Reduce the application specific reduction variables. If no value
  // was computed on an MPI rank, a benign value will be set. If the
  // reduction result is also a benign value, that means no MPI rank
  // wants to change the value and it will be ignored (if a double).
  for ( auto & var : m_appReductionVars )
    var.second->reduce( new_dw );

  // Specific handling for double reduction vars.
  if( m_appReductionVars[outputInterval_name]->active )
  {
    if( !m_appReductionVars[outputInterval_name]->min_var.isBenignValue() )
      m_output->setOutputInterval( m_appReductionVars[outputInterval_name]->min_var );
  }

  if( m_appReductionVars[checkpointInterval_name]->active )
  {
    if( !m_appReductionVars[checkpointInterval_name]->min_var.isBenignValue() )
      m_output->setCheckpointInterval( m_appReductionVars[checkpointInterval_name]->min_var );
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
  // std::cerr << "**********  " << __FUNCTION__ << "  " << __LINE__ << "  "
  //        << new_dw << std::endl;  

  // Initialize the time step.
  new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);

  // m_materialManager->setCurrentTopLevelTimeStep( m_timeStep );  

  // Initialize the simulation time.
  new_dw->put(simTime_vartype(m_simTime), m_simulationTimeLabel);

  // m_materialManager->setElapsedSimTime( m_simTime );

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
  // If recomuting a time step do not update the time step or the
  // simulation time.
  bool val;
  if ( !getReductionVariable( recomputeTimeStep_name, val ) ) {
    // Store the time step so it can be incremented at the top of the
    // time step where it is over written.
    new_dw->put(timeStep_vartype(m_timeStep), m_timeStepLabel);
    
    // Update the simulation time.
    m_simTime += m_delT;

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

  proc0cout << "WARNING Recomputng time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
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

  // When recomputing the delT, rank 0 determines the value and
  // sends it to all other ranks.
  Uintah::MPI::Bcast( &new_delT, 1, MPI_DOUBLE, 0, d_myworld->getComm() );
    
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

  // Clear the time step based reduction variables.
  clearReductionVariables();
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
    std::ostringstream message;

    message << "WARNING (a) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": raising the next delT from " << m_nextDelT;
    
    m_nextDelT = m_simulationTime->m_delt_min;
    
    message << " to minimum: " << m_nextDelT;

    DOUT( d_myworld->myRank() == 0 && g_deltaT_major_warnings, message.str() );
  }

  // Check to see if the next delT was increased too much over the
  // current delT
  double delt_tmp = (1.0+m_simulationTime->m_max_delt_increase) * m_delT;
  
  if( m_delT > 0.0 &&
      m_simulationTime->m_max_delt_increase > 0 &&
      m_nextDelT > delt_tmp ) {
    std::ostringstream message;

    message << "WARNING (b) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": lowering the next delT from " << m_nextDelT;
    
    m_nextDelT = delt_tmp;
    
    message << " to maxmimum: " << m_nextDelT
              << " (maximum increase of " << m_simulationTime->m_max_delt_increase
              << ")";

    DOUT( d_myworld->myRank() == 0 && g_deltaT_major_warnings, message.str() );
  }

  // Check to see if the next delT exceeds the max_initial_delt
  if( m_simulationTime->m_max_initial_delt > 0 &&
      m_simTime+m_delT <= m_simulationTime->m_initial_delt_range &&
      m_nextDelT > m_simulationTime->m_max_initial_delt ) {
    std::ostringstream message;

    message << "WARNING (c) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": lowering the next delT from " << m_nextDelT ;

    m_nextDelT = m_simulationTime->m_max_initial_delt;

    message<< " to maximum: " << m_nextDelT
             << " (for initial timesteps)";

    DOUT( d_myworld->myRank() == 0 && g_deltaT_major_warnings, message.str() );
  }

  // Check to see if the next delT exceeds the delt_max
  if( m_nextDelT > m_simulationTime->m_delt_max ) {
    std::ostringstream message;

    message << "WARNING (d) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": lowering the next delT from " << m_nextDelT;

    m_nextDelT = m_simulationTime->m_delt_max;
    
    message << " to maximum: " << m_nextDelT;

    DOUT( d_myworld->myRank() == 0 && g_deltaT_minor_warnings, message.str() );
  }

  // Clamp the next delT to match the requested output and/or
  // checkpoint times
  if( m_simulationTime->m_clamp_time_to_output ) {

    // Clamp to the output time
    double nextOutput = m_output->getNextOutputTime();
    if (nextOutput != 0 && m_simTime + m_delT + m_nextDelT > nextOutput) {
      std::ostringstream message;

      message << "WARNING (e) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": lowering the next delT from " << m_nextDelT;

      m_nextDelT = nextOutput - (m_simTime+m_delT);

      message << " to " << m_nextDelT
                << " to line up with output time";

      DOUT( d_myworld->myRank() == 0 && g_deltaT_minor_warnings, message.str() );
    }

    // Clamp to the checkpoint time
    double nextCheckpoint = m_output->getNextCheckpointTime();
    if (nextCheckpoint != 0 && m_simTime + m_delT + m_nextDelT > nextCheckpoint) {
      std::ostringstream message;
      
      message << "WARNING (f) at time step " << m_timeStep << " "
            << "and sim time " << m_simTime << " "
            << ": lowering the next delT from " << m_nextDelT;

      m_nextDelT = nextCheckpoint - (m_simTime+m_delT);

      message << " to " << m_nextDelT
                << " to line up with checkpoint time";

      DOUT( d_myworld->myRank() == 0 && g_deltaT_minor_warnings, message.str() );
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
ApplicationCommon::isLastTimeStep( double walltime )
{
  bool val;

  if( getReductionVariable( endSimulation_name, val ) )
    return true;
  
  if( getReductionVariable( abortTimeStep_name, val ) )
    return true;

  if( m_simTime >= m_simulationTime->m_max_time )
    return true;

  if( m_simulationTime->m_max_time_steps > 0 &&
      m_timeStep >= m_simulationTime->m_max_time_steps )
    return true;

  if( m_simulationTime->m_max_wall_time > 0 )
  {
    // When using the wall clock time, rank 0 determines the time and
    // sends it to all other ranks.
    Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );

    if(walltime >= m_simulationTime->m_max_wall_time )
      return true;
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
  if( m_simTime + m_delT >= m_simulationTime->m_max_time )
    return true;
           
  if( m_simulationTime->m_max_time_steps > 0 &&
      m_timeStep + 1 >= m_simulationTime->m_max_time_steps )
    return true;
           
  if( m_simulationTime->m_max_wall_time > 0 )
  {
    // When using the wall clock time, rank 0 determines the time and
    // sends it to all other ranks.
    Uintah::MPI::Bcast( &walltime, 1, MPI_DOUBLE, 0, d_myworld->getComm() );
  
    if( walltime >= m_simulationTime->m_max_wall_time )
      return true;
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

  // Write the time step to the inital DW so apps can get to it when
  // scheduling.
  m_scheduler->getLastDW()->override(timeStep_vartype(m_timeStep),
                                     m_timeStepLabel );
  
  // m_materialManager->setCurrentTopLevelTimeStep( m_timeStep );
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

  // m_materialManager->setCurrentTopLevelTimeStep( m_timeStep );
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
  
  // m_materialManager->setElapsedSimTime( m_simTime );
}

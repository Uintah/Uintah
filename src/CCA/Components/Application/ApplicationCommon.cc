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
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Task.h>

using namespace Uintah;

ApplicationCommon::ApplicationCommon()
{
  m_AMR                       = false;
  m_lockstepAMR               = false;
  
  m_dynamicRegridding         = false;
  
  m_haveModifiedVars          = false;
  
  m_updateCheckpointInterval  = false;
  m_updateOutputInterval      = false;

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
  VarLabel::destroy(m_outputIntervalLabel);
  VarLabel::destroy(m_outputTimestepIntervalLabel);
  VarLabel::destroy(m_checkpointIntervalLabel);
  VarLabel::destroy(m_checkpointTimestepIntervalLabel);
}

void ApplicationCommon::problemSetup( const ProblemSpecP &prob_spec )
{
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

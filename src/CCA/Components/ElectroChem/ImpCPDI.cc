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

#include <CCA/Components/ElectroChem/ImpCPDI.h>
#include <Core/Grid/SimulationState.h>

using namespace Uintah;


ImpCPDI::ImpCPDI(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  d_mpm_lb = scinew MPMLabel();
  d_impec_flags = scinew ImpECFlags();

  d_SMALL_NUM = 1e-200;
  d_initial_dt = 0.0;
  d_next_output_time = 0.0;
  d_stop_time = 0.0;
  d_num_iterations = 0;
  d_NGP = 1;
  d_NGN = 1;

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();
}

ImpCPDI::~ImpCPDI()
{
  delete d_mpm_lb;
  delete d_impec_flags;

  if(d_one_matl->removeReference()){
    delete d_one_matl;
  }
}

void ImpCPDI::problemSetup( const ProblemSpecP&     params,
                            const ProblemSpecP&     restart_prob_spec,
                                  GridP&            grid,
                                  SimulationStateP& state )
{
  d_shared_state = state;
}

void ImpCPDI::preGridProblemSetup( const ProblemSpecP&     params, 
                                         GridP&            grid,
                                         SimulationStateP& state )
{
}

void ImpCPDI::outputProblemSpec( ProblemSpecP& ps )
{
}
      
void ImpCPDI::scheduleInitialize( const LevelP&     level,
                                        SchedulerP& sched )
{
}
                                 
void ImpCPDI::scheduleRestartInitialize( const LevelP&     level,
                                               SchedulerP& sched )
{
}

void ImpCPDI::scheduleComputeStableTimestep( const LevelP&     level,
                                                   SchedulerP& sched )
{
}
      
void ImpCPDI::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
}

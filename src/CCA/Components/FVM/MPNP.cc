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


#include <CCA/Components/FVM/MPNP.h>
#include <CCA/Components/FVM/FVMBoundCond.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>

using namespace Uintah;

MPNP::MPNP(const ProcessorGroup* myworld,
	   const MaterialManagerP materialManager)
  : ApplicationCommon(myworld, materialManager)
{
  d_lb = scinew FVMLabel();

  d_delt = 0;
  d_solver = 0;

  d_mpnp_matl  = scinew MaterialSubset();
  d_mpnp_matl->add(0);
  d_mpnp_matl->addReference();

  d_mpnp_matlset  = scinew MaterialSet();
  d_mpnp_matlset->add(0);
  d_mpnp_matlset->addReference();

  std::cout << "MPNP Constructor" << std::endl;
}
//__________________________________
//
MPNP::~MPNP()
{
  delete d_lb;

  if (d_mpnp_matl && d_mpnp_matl->removeReference()){
    delete d_mpnp_matl;
  }

  if (d_mpnp_matlset && d_mpnp_matlset->removeReference()){
    delete d_mpnp_matlset;
  }
}
//__________________________________
//
void MPNP::problemSetup(const ProblemSpecP& prob_spec,
			const ProblemSpecP& restart_prob_spec,
			GridP& grid)
{
  ProblemSpecP root_ps = 0;
  if (restart_prob_spec){
    root_ps = restart_prob_spec;
  } else{
    root_ps = prob_spec;
  }
}

void
MPNP::outputProblemSpec(ProblemSpecP& ps)
{
}

//__________________________________
// 
void
MPNP::scheduleInitialize( const LevelP     & level,
                                              SchedulerP & sched )
{
}

//__________________________________
//
void MPNP::scheduleRestartInitialize(const LevelP& level,
                                            SchedulerP& sched)
{
}

//__________________________________
// 
void MPNP::scheduleComputeStableTimeStep(const LevelP& level,
                                          SchedulerP& sched)
{

}

//__________________________________
//
void
MPNP::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{

}

//__________________________________
//
void MPNP::computeStableTimeStep(const ProcessorGroup*,
                                  const PatchSubset* pss,
                                  const MaterialSubset*,
                                  DataWarehouse*, DataWarehouse* new_dw)
{
}

//__________________________________
//
void MPNP::initialize(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*, DataWarehouse* new_dw)
{
}

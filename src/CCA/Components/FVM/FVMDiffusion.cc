/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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


#include <CCA/Components/FVM/FVMDiffusion.h>
#include <CCA/Components/FVM/FVMMaterial.h>
#include <Core/Labels/FVMLabel.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
using namespace std;
using namespace Uintah;

FVMDiffusion::FVMDiffusion(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
	lb = scinew FVMLabel();
}

FVMDiffusion::~FVMDiffusion()
{
  delete lb;
}

void FVMDiffusion::problemSetup(const ProblemSpecP& prob_spec,
                            const ProblemSpecP& restart_prob_spec,
                            GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;

	//Finding FVM block in problem spec
	ProblemSpecP restart_fvm_ps = 0;
	ProblemSpecP fvm_ps = prob_spec->findBlock("FVM");
	if(fvm_ps)
		restart_fvm_ps = fvm_ps;
	else
		restart_fvm_ps = restart_prob_spec->findBlock("FVM");
	
	// Getting timestep
	restart_fvm_ps->require("delt", delt_);

  ProblemSpecP restart_mat_ps = 0;
  ProblemSpecP mat_ps = 
    prob_spec->findBlockWithOutAttribute("MaterialProperties");

	// Testing for restart. !!!!Find out why this is needed!!!!
  if (mat_ps)
    restart_mat_ps = mat_ps;
  else if (restart_prob_spec)
    restart_mat_ps =
			restart_prob_spec->findBlockWithOutAttribute("MaterialProperties");

	//Iterate through different fvm materials and add to sharedState
  ProblemSpecP fvm_mat_ps = restart_mat_ps->findBlock("FVM");
  ProblemSpecP ps = fvm_mat_ps->findBlock("material");
	for(ProblemSpecP ps = fvm_mat_ps->findBlock("material"); ps != 0;
			ps = ps->findNextBlock("material")){
  	FVMMaterial* mat = scinew FVMMaterial(ps, sharedState);
  	sharedState->registerFVMMaterial(mat);
	}
}
 
void FVMDiffusion::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &FVMDiffusion::initialize);
  task->computes(lb->concentration_CCLabel);
  sched->addTask(task, level->eachPatch(), sharedState_->allFVMMaterials());
	cout << "Doing Schedule Initialize" << endl;
}
 
void FVMDiffusion::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &FVMDiffusion::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void FVMDiffusion::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &FVMDiffusion::timeAdvance);

  task->requires(Task::OldDW, lb->concentration_CCLabel, Ghost::AroundNodes, 1);
  task->computes(lb->concentration_CCLabel);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

}

void FVMDiffusion::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

void FVMDiffusion::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
			FVMMaterial* fvm_matl = sharedState_->getFVMMaterial(m);
      int index = fvm_matl->getDWIndex();

      CCVariable<double> concentration;
      new_dw->allocateAndPut(concentration, lb->concentration_CCLabel, index, patch);
      concentration.initialize(fvm_matl->getConcentration());

      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
				IntVector l,h;
				patch->getFaceNodes(Patch::xminus, 0, l, h);

			for(NodeIterator iter(l,h); !iter.done(); iter++)
	  		concentration[*iter]=1;
      }
			for(CellIterator iter = patch->getCellIterator(); !iter.done(); ++iter){
				IntVector n = *iter;
				cout << n << concentration[n] <<endl;
			}
    }
  }

	cout << "Initialized Grid" << endl;
}

void FVMDiffusion::timeAdvance(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
    }
  }
}

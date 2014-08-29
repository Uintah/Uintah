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
#include <CCA/Ports/LoadBalancer.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
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
  concentration_label = VarLabel::create("g.concentration", 
                               NCVariable<double>::getTypeDescription());
}

FVMDiffusion::~FVMDiffusion()
{
  VarLabel::destroy(concentration_label);
}

void FVMDiffusion::problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& restart_prob_spec,
                            GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP diffspec = params->findBlock("FVMDiffusion");
  diffspec->require("diffusivity", diffusivity);
	diffspec->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
	cout << "diffusivity is: " << diffusivity << endl;
	cout << "delt is: " << delt_ << endl;
}
 
void FVMDiffusion::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &FVMDiffusion::initialize);
  task->computes(concentration_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
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

  task->requires(Task::OldDW, concentration_label, Ghost::AroundNodes, 1);
  task->computes(concentration_label);
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
		       DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      NCVariable<double> concentration;
      new_dw->allocateAndPut(concentration, concentration_label, matl, patch);
      concentration.initialize(0);

      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
	IntVector l,h;
	patch->getFaceNodes(Patch::xminus, 0, l, h);

	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  concentration[*iter]=1;
      }
    }
  }
}

void FVMDiffusion::timeAdvance(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      constNCVariable<double> temperature;

      old_dw->get(temperature, concentration_label, matl, patch, 
                  Ghost::AroundNodes, 1);

      NCVariable<double> newtemperature;

      new_dw->allocateAndPut(newtemperature, concentration_label, matl, patch);
      newtemperature.copyPatch(temperature, newtemperature.getLow(), 
                               newtemperature.getHigh());

      double residual=0;
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex(); 

      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);

      delt_vartype dt;
      old_dw->get(dt, sharedState_->get_delt_label());
      Vector dx = patch->getLevel()->dCell();
      Vector diffusion_number(1./(dx.x()*dx.x()), 1./(dx.y()*dx.y()),
                              1./(dx.z()*dx.z()));
      
      double k = .5;

      cout << "dx = " << dx << endl;
      diffusion_number = diffusion_number* k*dt;
      cout << "diffusion_number = " << diffusion_number << endl;

      for(NodeIterator iter(l, h);!iter.done(); iter++){
	newtemperature[*iter]=(1./6)*(
	  temperature[*iter+IntVector(1,0,0)]+temperature[*iter+IntVector(-1,0,0)]+
	  temperature[*iter+IntVector(0,1,0)]+temperature[*iter+IntVector(0,-1,0)]+
	  temperature[*iter+IntVector(0,0,1)]+temperature[*iter+IntVector(0,0,-1)]);
	double diff = newtemperature[*iter]-temperature[*iter];
	residual += diff*diff;
      }
    }
  }
}

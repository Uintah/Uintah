
#include <Packages/Uintah/CCA/Components/Examples/Burger.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

Burger::Burger(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

Burger::~Burger()
{
  delete lb_;
}

void Burger::problemSetup(const ProblemSpecP& params, GridP& /*grid*/,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP burger = params->findBlock("Burger");
  burger->require("delt", delt_);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void Burger::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Burger::initialize);
  task->computes(lb_->u);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void Burger::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Burger::computeStableTimestep);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
Burger::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched, int, int )
{
  Task* task = scinew Task("timeAdvance",
			   this, &Burger::timeAdvance);
  task->requires(Task::OldDW, lb_->u, Ghost::AroundNodes, 1);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->computes(lb_->u);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Burger::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset*,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label());
}

void Burger::initialize(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<double> u;
      new_dw->allocateAndPut(u, lb_->u, matl, patch);
      //Initialize according to the function
      // u = sin( pi*x ) + sin( pi*2*y )
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();
      for( NodeIterator iter(l,h); !iter.done(); iter++ ){
        Point p = patch->nodePosition(*iter);
        u[*iter] = sin( p.x() * 3.14159265358 ) + sin( p.y() * 2*3.14159265358 );
      }
      // allocateAndPut instead:
      /* new_dw->put(u, lb_->u, matl, patch); */;
    }
  }
}

void Burger::timeAdvance(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  //Loop for all patches on this processor
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

     delt_vartype dt;
     old_dw->get(dt, sharedState_->get_delt_label());

     constNCVariable<double> u;

      // Pull the data off the shelf for the old data warehouse, including 1 layer of
      // "ghost" nodes from the surrounding patches
      old_dw->get(u, lb_->u, matl, patch, Ghost::AroundNodes, 1);

      // Also get dt, dx, and dy
      old_dw->get(dt, sharedState_->get_delt_label());

      NCVariable<double> newu;
      new_dw->allocateAndPut(newu, lb_->u, matl, patch);
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex(); 

      //On true edges, stay one node inside the patch
      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                     0 );
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
                     0 );

      //Do some invariant calculations outside of loop:
      Vector v = patch->getLevel()->dCell();
      double dt_div_twodx, dt_div_twody;
      dt_div_twodx = dt/(2*v.x());
      dt_div_twody = dt/(2*v.y());

      //Iterate through all the nodes and assign the new values
      for(NodeIterator iter(l, h);!iter.done(); iter++){
	double dudx_mult_dt = (u[*iter+IntVector(1,0,0)] - u[*iter-IntVector(1,0,0)]) * dt_div_twodx;
	double dudy_mult_dt = (u[*iter+IntVector(0,1,0)] - u[*iter-IntVector(0,1,0)]) * dt_div_twody;
	double du = - u[*iter] * (dudx_mult_dt + dudy_mult_dt);
	newu[*iter]= u[*iter] + du;
      }

      //Make sure the trailing edges are updated to the same value as the adjacent interior
      // point to prevent the creation of discontinuities
      if (patch->getBCType(Patch::xplus) != Patch::Neighbor) {
        l = patch->getNodeLowIndex();
        h = patch->getNodeHighIndex();
        l.x( h.x()-1 );
        for (NodeIterator iter(l, h);!iter.done(); iter++){
          newu[*iter] = newu[*iter-IntVector(1,0,0)];
        }
      }
      if (patch->getBCType(Patch::yplus) != Patch::Neighbor) {
        l = patch->getNodeLowIndex();
        h = patch->getNodeHighIndex();
        l.y( h.y()-1 );
        for (NodeIterator iter(l, h);!iter.done(); iter++){
          newu[*iter] = newu[*iter-IntVector(0,1,0)];
        }
      }
      if (patch->getBCType(Patch::xminus) != Patch::Neighbor) {
        l = patch->getNodeLowIndex();
        h = patch->getNodeHighIndex();
        h.x( l.x()+1 );
        for (NodeIterator iter(l, h);!iter.done(); iter++){
          newu[*iter] = u[*iter];
        }
      }
      if (patch->getBCType(Patch::yminus) != Patch::Neighbor) {
        l = patch->getNodeLowIndex();
        h = patch->getNodeHighIndex();
        h.y( l.y()+1 );
        for (NodeIterator iter(l, h);!iter.done(); iter++){
          newu[*iter] = u[*iter];
        }
      }

      // Store all the new values calculated into the new data warehouse
      // allocateAndPut instead:
      /* new_dw->put(newu, lb_->u, matl, patch); */;
    }
  }
}

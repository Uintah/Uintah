
#include <Packages/Uintah/CCA/Components/Examples/SolverTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SolverTest1::SolverTest1(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

SolverTest1::~SolverTest1()
{
  delete lb_;
}

void SolverTest1::problemSetup(const ProblemSpecP& prob_spec, GridP&,
			 SimulationStateP& sharedState)
{
  solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver) {
    throw InternalError("ST1:couldn't get solver port");
  }
  // TODO - get SolverParameters?
  // I copied this out of ICE.cc
  // Pull out implicit solver parameters
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  ProblemSpecP cfd_st_ps = cfd_ps->findBlock("SolverTest");
  
  ProblemSpecP impSolver = cfd_st_ps->findBlock("ImplicitSolver");
  if (impSolver) {
    //d_delT_knob = 0.5;      // default value when running implicit
    solver_parameters = solver->readParameters(impSolver, "implicitPressure");
    solver_parameters->setSolveOnExtraCells(false);
  }
  sharedState_ = sharedState;
  ProblemSpecP ST = prob_spec->findBlock("SolverTest");
  ST->require("delt", delt_);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void SolverTest1::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &SolverTest1::initialize);

  task->computes(lb_->pressure);
  task->computes(lb_->density);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void SolverTest1::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &SolverTest1::computeStableTimestep);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
SolverTest1::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched,
			       int, int )
{
  Task* task = scinew Task("timeAdvance",
			   this, &SolverTest1::timeAdvance,
			   level, sched.get_rep());
  task->hasSubScheduler();
  task->requires(Task::OldDW, lb_->pressure, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->density, Ghost::None, 0);

  task->computes(lb_->density);
  task->computes(lb_->pressure_matrix);
  task->computes(lb_->pressure_rhs);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  solver->scheduleSolve(level, sched, sharedState_->allMaterials(), lb_->pressure_matrix, 
    Task::NewDW, lb_->pressure, false, lb_->pressure_rhs, Task::NewDW, 0, Task::OldDW, solver_parameters);

}

void SolverTest1::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset*,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label());
}

void SolverTest1::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double> pressure;
      CCVariable<double> density;
      new_dw->allocateAndPut(pressure, lb_->pressure, matl, patch);
      new_dw->allocateAndPut(density, lb_->density, matl, patch);

      // do something with pressure and density
      pressure.initialize(0);
      density.initialize(0);
    }
  }
}

void SolverTest1::timeAdvance(const ProcessorGroup* pg,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw, DataWarehouse* new_dw,
			   LevelP level, Scheduler* sched)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> old_pressure;
      constCCVariable<double> old_density;
      old_dw->get(old_pressure, lb_->pressure, matl, patch, Ghost::None, 0);
      old_dw->get(old_density, lb_->density, matl, patch, Ghost::None, 0);

      CCVariable<Stencil7> pressure_matrix;
      CCVariable<double> pressure_rhs;
      CCVariable<double> density;
      new_dw->allocateAndPut(pressure_matrix, lb_->pressure_matrix, matl, patch);
      new_dw->allocateAndPut(pressure_rhs, lb_->pressure_rhs, matl, patch);
      new_dw->allocateAndPut(density, lb_->density, matl, patch);

      // make new density and pressure matrix and rhs to prep for pressure solve
    }
  }

}

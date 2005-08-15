
#include <Packages/Uintah/CCA/Components/Examples/SolverTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
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
    throw InternalError("ST1:couldn't get solver port", __FILE__, __LINE__);
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

  // whether or not to do laplacian in x,y,or z direction
  if (ST->findBlock("X_Laplacian"))
    x_laplacian = true;
  else
    x_laplacian = false;
  if (ST->findBlock("Y_Laplacian"))
    y_laplacian = true;
  else
    y_laplacian = false;
  if (ST->findBlock("Z_Laplacian"))
    z_laplacian = true;
  else
    z_laplacian = false;

  if (!x_laplacian && !y_laplacian && !z_laplacian)
    throw ProblemSetupException("SolverTest: Must specify one of X_Laplacian, Y_Laplacian, or Z_Laplacian",
                                __FILE__, __LINE__);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void SolverTest1::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
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
}

double rand_double()
{
  return ((double)rand())/RAND_MAX*10.0;
}

void SolverTest1::timeAdvance(const ProcessorGroup* pg,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw, DataWarehouse* new_dw,
			   LevelP level, Scheduler* sched)
{
  static int time = 0;
  time += (int) delt_*100;
  srand(time);

  int center = 0;
  int n=0, s=0, e=0, w=0, t=0, b=0;

  if (x_laplacian) {
    center+=2;
    e = -1;
    w = -1;
  }
  if (y_laplacian) {
    center+=2;
    n = -1;
    s = -1;
  }
  if (z_laplacian) {
    center+=2;
    t = -1;
    b = -1;
  }


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<Stencil7> A;
      CCVariable<double> rhs;
      new_dw->allocateAndPut(A, lb_->pressure_matrix, matl, patch);
      new_dw->allocateAndPut(rhs, lb_->pressure_rhs, matl, patch);

      bool first = true;
      for(CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
        IntVector c = *iter;
        Stencil7&  A_tmp=A[c];
        A_tmp.p = center; 
        A_tmp.n = n;   A_tmp.s = s;
        A_tmp.e = e;   A_tmp.w = w; 
        A_tmp.t = t;   A_tmp.b = b;
        
        if (c == IntVector(0,0,0)) {
          rhs[c] = 1.0;
        }
        else
          rhs[c] = 0;
      }

    }
  }

}

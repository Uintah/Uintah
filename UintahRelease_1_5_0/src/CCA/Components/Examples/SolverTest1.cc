/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <CCA/Components/Examples/SolverTest1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

SolverTest1::SolverTest1(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}
//__________________________________
//
SolverTest1::~SolverTest1()
{
  delete lb_;
  delete solver_parameters;
}
//__________________________________
//
void SolverTest1::problemSetup(const ProblemSpecP& prob_spec, 
                               const ProblemSpecP& restart_prob_spec, 
                               GridP&, SimulationStateP& sharedState)
{
  solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver) {
    throw InternalError("ST1:couldn't get solver port", __FILE__, __LINE__);
  }
  
  ProblemSpecP st_ps = prob_spec->findBlock("SolverTest");
  solver_parameters = solver->readParameters(st_ps, "implicitPressure",
                                             sharedState);
  solver_parameters->setSolveOnExtraCells(false);
    
  sharedState_ = sharedState;
  st_ps->require("delt", delt_);

  // whether or not to do laplacian in x,y,or z direction
  if (st_ps->findBlock("X_Laplacian"))
    x_laplacian = true;
  else
    x_laplacian = false;
  if (st_ps->findBlock("Y_Laplacian"))
    y_laplacian = true;
  else
    y_laplacian = false;
  if (st_ps->findBlock("Z_Laplacian"))
    z_laplacian = true;
  else
    z_laplacian = false;

  if (!x_laplacian && !y_laplacian && !z_laplacian)
    throw ProblemSetupException("SolverTest: Must specify one of X_Laplacian, Y_Laplacian, or Z_Laplacian",
                                __FILE__, __LINE__);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
//__________________________________
// 
void SolverTest1::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  solver->scheduleInitialize(level,sched,sharedState_->allMaterials());
}
//__________________________________
// 
void SolverTest1::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",this, 
                           &SolverTest1::computeStableTimestep);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//__________________________________
//
void
SolverTest1::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &SolverTest1::timeAdvance,
			   level, sched.get_rep());
  task->computes(lb_->pressure_matrix);
  task->computes(lb_->pressure_rhs);

  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  solver->scheduleSolve(level, sched, sharedState_->allMaterials(), 
                        lb_->pressure_matrix, Task::NewDW, lb_->pressure, 
                        false, lb_->pressure_rhs, Task::NewDW, 0, Task::OldDW, 
                        solver_parameters,false);

}
//__________________________________
//
void SolverTest1::computeStableTimestep(const ProcessorGroup*,
				  const PatchSubset* pss,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(pss));
}
//__________________________________
//
void SolverTest1::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw)
{
}
//______________________________________________________________________
//
void SolverTest1::timeAdvance(const ProcessorGroup* pg,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw, DataWarehouse* new_dw,
			   LevelP level, Scheduler* sched)
{
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
      new_dw->allocateAndPut(A,   lb_->pressure_matrix, matl, patch);
      new_dw->allocateAndPut(rhs, lb_->pressure_rhs,    matl, patch);

      //bool first = true;
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

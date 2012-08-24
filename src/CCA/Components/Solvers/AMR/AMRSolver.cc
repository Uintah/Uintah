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


#include <sci_defs/hypre_defs.h>
#include <CCA/Components/Solvers/AMR/AMRSolver.h>
#include <CCA/Components/Solvers/AMR/HypreDriver.h>
#include <CCA/Components/Solvers/HypreSolverParams.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static DebugStream cout_dbg("HYPRE_DBG", false);


AMRSolver::AMRSolver(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld) {}
  
AMRSolver::~AMRSolver() {}


/*_____________________________________________________________________
 Function AMRSolver::readParameters 
 _____________________________________________________________________*/
SolverParameters*
AMRSolver::readParameters(ProblemSpecP& params,
                          const string& varname,
                          SimulationStateP& state)
  
{
  HypreSolverParams* p = new HypreSolverParams();
  bool found=false;

  /* Scan and set parameters */
  if(params){
    for(ProblemSpecP param = params->findBlock("Parameters"); param != 0;
        param = param->findNextBlock("Parameters")) {
      string variable;
      if(param->getAttribute("variable", variable) && variable != varname)
        continue;
      param->getWithDefault("solver", p->solverTitle, "smg");
      param->getWithDefault("preconditioner", p->precondTitle, "diagonal");
      param->getWithDefault("tolerance", p->tolerance, 1.e-10);
      param->getWithDefault("maxiterations", p->maxIterations, 75);
      param->getWithDefault("npre", p->nPre, 1);
      param->getWithDefault("npost", p->nPost, 1);
      param->getWithDefault("skip", p->skip, 0);
      param->getWithDefault("jump", p->jump, 0);
      param->getWithDefault("logging", p->logging, 0);
      param->getWithDefault("outputEquations", p->printSystem,false);
      found=true;
    }
  }

  /* Default parameter values */
  if(!found){
    p->solverTitle = "smg";
    p->precondTitle = "diagonal";
    p->tolerance = 1.e-10;
    p->maxIterations = 75;
    p->nPre = 1;
    p->nPost = 1;
    p->skip = 0;
    p->jump = 0;
    p->logging = 0;
  }
  p->symmetric = false;
  //  p->symmetric=true;
  p->restart=false;
  //  p->restart=true;

  return p;
}

SolverParameters*
AMRSolver::readParameters(ProblemSpecP& params,const string& varname)
{
  HypreSolverParams* p = new HypreSolverParams();
  bool found=false;

  /* Scan and set parameters */
  if(params){
    for(ProblemSpecP param = params->findBlock("Parameters"); param != 0;
        param = param->findNextBlock("Parameters")) {
      string variable;
      if(param->getAttribute("variable", variable) && variable != varname)
        continue;
      param->getWithDefault("solver", p->solverTitle, "smg");
      param->getWithDefault("preconditioner", p->precondTitle, "diagonal");
      param->getWithDefault("tolerance", p->tolerance, 1.e-10);
      param->getWithDefault("maxiterations", p->maxIterations, 75);
      param->getWithDefault("npre", p->nPre, 1);
      param->getWithDefault("npost", p->nPost, 1);
      param->getWithDefault("skip", p->skip, 0);
      param->getWithDefault("jump", p->jump, 0);
      param->getWithDefault("logging", p->logging, 0);
      param->getWithDefault("outputEquations", p->printSystem,false);
      found=true;
    }
  }

  /* Default parameter values */
  if(!found){
    p->solverTitle = "smg";
    p->precondTitle = "diagonal";
    p->tolerance = 1.e-10;
    p->maxIterations = 75;
    p->nPre = 1;
    p->nPost = 1;
    p->skip = 0;
    p->jump = 0;
    p->logging = 0;
  }
  p->symmetric = false;
  //  p->symmetric=true;
  p->restart=false;
  //  p->restart=true;

  return p;
}

//______________________________________________________________________
//  This originated from Steve's implementation of HypreSolver
void
AMRSolver::scheduleSolve(const LevelP& level, SchedulerP& sched,
                         const MaterialSet* matls,
                         const VarLabel* A,       Task::WhichDW which_A_dw,  
                         const VarLabel* x,       bool modifies_x,
                         const VarLabel* b,       Task::WhichDW which_b_dw,  
                         const VarLabel* guess,   Task::WhichDW which_guess_dw,
                         const SolverParameters* params,
                         bool modifies_hypre)
  
{
  cout_doing << "AMRSolver::scheduleSolve() BEGIN" << "\n";
  Task* task;

  ASSERTEQ(A->typeDescription()->getType(), x->typeDescription()->getType());
  ASSERTEQ(A->typeDescription()->getType(), b->typeDescription()->getType());
  const HypreSolverParams* dparams = dynamic_cast<const HypreSolverParams*>(params);
  if(!dparams)
    throw InternalError("Wrong type of params passed to hypre solver!",
                        __FILE__, __LINE__);

  /* Decide which Hypre interface to use */
  HypreInterface interface;
  int numLevels = level->getGrid()->numLevels();
  if (numLevels > 1) {
    interface = HypreSStruct;   /* Composite grid of uniform patches */
  } else {
    interface = HypreSStruct;   /* A uniform grid */
  }

  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perProcPatches = lb->getPerProcessorPatchSet(level->getGrid());
  
  HypreDriver* that = newHypreDriver(interface,level.get_rep(), matls, A, which_A_dw,x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams, perProcPatches);
  Handle<HypreDriver > handle = that;

  void (HypreDriver::*func)(const ProcessorGroup*, const PatchSubset*,
                            const MaterialSubset*,
                            DataWarehouse*, DataWarehouse*,
                            Handle<HypreDriver>);
                            
  func = &HypreDriver::solve<CCTypes>;
  task = scinew Task("AMRSolver::Matrix solve CC", that, func, handle);
      
  //__________________________________
  // computes and requires for A, X and rhs
  for (int i = 0; i < level->getGrid()->numLevels(); i++) {
    const LevelP l = level->getGrid()->getLevel(i);
    const PatchSubset* subset = l->eachPatch()->getUnion();
    
    task->requires(which_A_dw, A, subset, Ghost::None, 0);
    
    if (modifies_x) {
      task->modifies(x, subset, 0);
    }
    else {
      task->computes(x, subset);
    }
    
    if (guess) {
      task->requires(which_guess_dw, guess, subset, Ghost::None, 0); 
    }
    
    task->requires(which_b_dw, b, subset, Ghost::None, 0);
  }// numLevels
  
  
  task->setType(Task::OncePerProc);

  sched->addTask(task, perProcPatches, matls);

  cout_doing << "AMRSolver::scheduleSolve() END" << "\n";
} // end scheduleSolve()


string AMRSolver::getName(){
  return "hypreamr";
}


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

#include <sci_defs/hypre_defs.h>

#include <CCA/Components/Solvers/AMR/AMRSolver.h>
#include <CCA/Components/Solvers/AMR/HypreDriver.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/StringUtil.h>
#include <iomanip>

using std::string;
using namespace Uintah;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "SOLVER_DOING_COUT:+"

static DebugStream cout_doing("SOLVER_DOING_COUT", false);
static DebugStream cout_dbg("SOLVER_DBG", false);


//______________________________________________________________________
//
AMRSolver::AMRSolver(const ProcessorGroup* myworld)
  : SolverCommon(myworld) 
{
  m_params = scinew HypreSolverParams();
}
  
AMRSolver::~AMRSolver() 
{
  delete m_params;
}


/*_____________________________________________________________________
 Function AMRSolver::readParameters 
 _____________________________________________________________________*/
void
AMRSolver::readParameters(ProblemSpecP& params_ps,
                          const string& varname)
{
  bool found=false;

  /* Scan and set parameters */
  if(params_ps){
    for( ProblemSpecP param_ps = params_ps->findBlock("Parameters"); param_ps != nullptr; param_ps = param_ps->findNextBlock("Parameters") ) {
      string variable;
      if( param_ps->getAttribute("variable", variable) && variable != varname ) {
        continue;
      }
      string str_solver;
      string str_precond;
      param_ps->getWithDefault("solver",          str_solver, "smg");
      param_ps->getWithDefault("preconditioner",  str_precond, "diagonal");
      param_ps->getWithDefault("tolerance",       m_params->tolerance, 1.e-10);
      param_ps->getWithDefault("maxiterations",   m_params->maxIterations, 75);
      param_ps->getWithDefault("npre",            m_params->nPre, 1);
      param_ps->getWithDefault("npost",           m_params->nPost, 1);
      param_ps->getWithDefault("skip",            m_params->skip, 0);
      param_ps->getWithDefault("jump",            m_params->jump, 0);
      param_ps->getWithDefault("logging",         m_params->logging, 0);
      param_ps->getWithDefault("outputEquations", m_params->printSystem,false);
      found=true;
      
      // convert to lower case
      m_params->solverTitle   = string_tolower( str_solver );
      m_params->precondTitle  = string_tolower( str_precond );
    }
  }

  /* Default parameter values */
  if( !found ){
    m_params->solverTitle = "smg";
    m_params->precondTitle = "diagonal";
    m_params->tolerance = 1.e-10;
    m_params->maxIterations = 75;
    m_params->nPre = 1;
    m_params->nPost = 1;
    m_params->skip = 0;
    m_params->jump = 0;
    m_params->logging = 0;
  }
  m_params->symmetric = false;
  //  m_params->symmetric=true;
  m_params->recompute=false;
  //  m_params->recompute=true;
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
                         bool isFirstSolve)
  
{
  cout_doing << "AMRSolver::scheduleSolve() BEGIN" << "\n";
  Task* task;

  ASSERTEQ(A->typeDescription()->getType(), x->typeDescription()->getType());
  ASSERTEQ(A->typeDescription()->getType(), b->typeDescription()->getType());

  /* Decide which Hypre interface to use */
  HypreInterface interface;
  int numLevels = level->getGrid()->numLevels();
  if (numLevels > 1) {
    interface = HypreSStruct;   /* Composite grid of uniform patches */
  } else {
    interface = HypreSStruct;   /* A uniform grid */
  }

  LoadBalancer * lb             = sched->getLoadBalancer();
  const PatchSet   * perProcPatches = lb->getPerProcessorPatchSet(level->getGrid());
  
  HypreDriver* that = newHypreDriver(interface,level.get_rep(), matls, A, which_A_dw,x, modifies_x, b, which_b_dw, guess, which_guess_dw, m_params, perProcPatches);
  Handle<HypreDriver > handle = that;

  void (HypreDriver::*func)(const ProcessorGroup*, 
                            const PatchSubset*,
                            const MaterialSubset*,
                            DataWarehouse*, 
                            DataWarehouse*,
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
  
  
  // task->computes( VarLabel::find(abortTimeStep_name) );
  // task->computes( VarLabel::find(recomputeTimeStep_name) );
  
  task->setType(Task::OncePerProc);

  sched->addTask(task, perProcPatches, matls);

  cout_doing << "AMRSolver::scheduleSolve() END" << "\n";
} // end scheduleSolve()


string AMRSolver::getName(){
  return "hypreamr";
}


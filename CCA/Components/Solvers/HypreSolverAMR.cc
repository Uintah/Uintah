/*--------------------------------------------------------------------------
 * File: HypreSolverAMR.cc
 *
 * Uintah wrap around HypreDriver class that schedules the solver call.
 * Here we read the solver parameters, so we when adding a new solver
 * or new variable type (NC, FC), remember to update the relevant functions
 * here.
 * See also HypreSolverAMR.h.
 *--------------------------------------------------------------------------*/
// TODO (taken from HypreSolver.cc):
// Matrix file - why are ghosts there?
// Read hypre options from input file
// 3D performance
// Logging?
// Report mflops
// Use a symmetric matrix
// More efficient set?
// Reuse some data between solves?

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverAMR.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;

namespace Uintah {

  /*_____________________________________________________________________
    class HypreSolverAMR implementation
    _____________________________________________________________________*/

  HypreSolverAMR::HypreSolverAMR(const ProcessorGroup* myworld)
    : UintahParallelComponent(myworld) {}
  
  HypreSolverAMR::~HypreSolverAMR() {}

  SolverParameters* HypreSolverAMR::readParameters(ProblemSpecP& params,
                                                   const string& varname)
    /*_____________________________________________________________________
      Function HypreSolverAMR::readParameters
      Load input parameters into the HypreSolverAMR parameter struct;
      check their correctness.
      _____________________________________________________________________*/
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
        param->getWithDefault("maxIterations", p->maxIterations, 75);
        param->getWithDefault("nPre", p->nPre, 1);
        param->getWithDefault("nPost", p->nPost, 1);
        param->getWithDefault("skip", p->skip, 0);
        param->getWithDefault("jump", p->jump, 0);
        param->getWithDefault("logging", p->logging, 0);
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
    p->symmetric=true; // TODO: this is currently turned off in AMR
                       // mode until we can support symmetric SStruct
                       // in the interface
    p->restart=true;

    return p;
  } // end readParameters()

  void
  HypreSolverAMR::scheduleSolve(const LevelP& level, SchedulerP& sched,
                                const MaterialSet* matls,
                                const VarLabel* A,
                                Task::WhichDW which_A_dw,  
                                const VarLabel* x,
                                bool modifies_x,
                                const VarLabel* b,
                                Task::WhichDW which_b_dw,  
                                const VarLabel* guess,
                                Task::WhichDW which_guess_dw,
                                const SolverParameters* params)
    /*_____________________________________________________________________
      Function HypreSolverAMR::scheduleSolve
      Create the Uintah task that solves the linear system using Hypre.
      We can accomodate different types of variables to be solved: CC,
      NC, FC, etc. NOTE: Currently only CC is supported.
      _____________________________________________________________________*/
  {
    Task* task;
    // The extra handle arg ensures that the stencil7 object will get freed
    // when the task gets freed.  The downside is that the refcount gets
    // tweaked everytime solve is called.

    TypeDescription::Type domtype = A->typeDescription()->getType();
    ASSERTEQ(domtype, x->typeDescription()->getType());
    ASSERTEQ(domtype, b->typeDescription()->getType());
    const HypreSolverParams* dparams =
      dynamic_cast<const HypreSolverParams*>(params);
    if(!dparams)
      throw InternalError("Wrong type of params passed to hypre solver!",
                          __FILE__, __LINE__);

    /* Decide which Hypre interface to use */
    HypreInterface interface;
    if (level->hasCoarserLevel() || level->hasFinerLevel()) {
      /* Composite grid of uniform patches */
      interface = HypreSStruct;
    } else {
      /* A uniform grid */
      interface = HypreStruct;
    }

    switch (domtype) {
    case TypeDescription::SFCXVariable:
    case TypeDescription::SFCYVariable:
    case TypeDescription::SFCZVariable:
    case TypeDescription::NCVariable:
      {
        throw InternalError("No supported solver for this variable type"
                            "in scheduleSolve", __FILE__, __LINE__);
      }
      
    case TypeDescription::CCVariable:
      {
        HypreDriver<CCTypes>* that = newHypreDriver<CCTypes>
          (interface,level.get_rep(), matls, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, dparams);
        Handle<HypreDriver<CCTypes> > handle = that;
        task = scinew Task("Matrix solve", that,
                           &HypreDriver<CCTypes>::solve, handle);
        break;
      } // case CCVariable

    default:
      throw InternalError("Unknown variable type in scheduleSolve",
                          __FILE__, __LINE__);
    }

    task->requires(which_A_dw, A, Ghost::None, 0);
    if (modifies_x) {
      task->modifies(x);
    }
    else {
      task->computes(x);
    }
    
    if (guess) {
      task->requires(which_guess_dw, guess, Ghost::None, 0); 
    }

    task->requires(which_b_dw, b, Ghost::None, 0);
    LoadBalancer* lb = sched->getLoadBalancer();
    sched->addTask(task, lb->createPerProcessorPatchSet(level), matls);
  } // end scheduleSolve()

} // end namespace Uintah

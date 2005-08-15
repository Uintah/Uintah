//--------------------------------------------------------------------------
// File: HyprePrecondSMG.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecondSMG.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSMG.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  void HyprePrecondSMG::setup(void)
  {
    if (_interface == HypreStruct) {
      HYPRE_StructSolver precond_solver_struct;
      HYPRE_StructSMGCreate(_pg->getComm(), &precond_solver_struct);
      HYPRE_StructSMGSetMemoryUse(precond_solver_struct, 0);
      HYPRE_StructSMGSetMaxIter(precond_solver_struct, 1);
      HYPRE_StructSMGSetTol(precond_solver_struct, 0.0);
      HYPRE_StructSMGSetZeroGuess(precond_solver_struct);
      HYPRE_StructSMGSetNumPreRelax(precond_solver_struct, _params->nPre);
      HYPRE_StructSMGSetNumPostRelax(precond_solver_struct, _params->nPost);
      HYPRE_StructSMGSetLogging(precond_solver_struct, 0);
      _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
      _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
      _precond_solver = (HYPRE_Solver) precond_solver_struct;
    }
  }

  void HyprePrecondSMG::destroy(void)
  {
    if (_interface == HypreStruct) {
      HYPRE_StructSMGDestroy((HYPRE_StructSolver) _precond_solver);
    }
  }

} // end namespace Uintah

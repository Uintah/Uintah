//--------------------------------------------------------------------------
// File: HyprePrecondPFMG.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecondPFMG.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondPFMG.h>
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

  HyprePrecondPFMG::HyprePrecondPFMG(const HypreInterface& interface,
                                     const ProcessorGroup* pg,
                                     const HypreSolverParams* params) :
    HyprePrecond(interface, pg, params,int(HypreStruct))
  {
    if (_interface == HypreStruct) {
      HYPRE_StructPFMGCreate(_pg->getComm(), &precond_solver);
      HYPRE_StructPFMGSetMaxIter(precond_solver, 1);
      HYPRE_StructPFMGSetTol(precond_solver, 0.0);
      HYPRE_StructPFMGSetZeroGuess(precond_solver);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(precond_solver, 1);
      HYPRE_StructPFMGSetNumPreRelax(precond_solver, params->nPre);
      HYPRE_StructPFMGSetNumPostRelax(precond_solver, params->nPost);
      HYPRE_StructPFMGSetSkipRelax(precond_solver, params->skip);
      HYPRE_StructPFMGSetLogging(precond_solver, 0);
      _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSolve;
      _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSetup;
      _precond_solver = (HYPRE_Solver) precond_solver;
    }
  }

  void HyprePrecondPFMG::~HyprePrecondPFMG(void)
  {
    if (_interface == HypreStruct) {
      HYPRE_StructPFMGDestroy((HYPRE_StructSolver) _precond_solver);
    }
  }

} // end namespace Uintah

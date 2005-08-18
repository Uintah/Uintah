//--------------------------------------------------------------------------
// File: HyprePrecondSMG.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecondSMG.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HyprePrecondSMG::initPriority(void)
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HyprePrecondSMG::setup(void)
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreSolverParams* params = driver->getParams();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSolver precond_solver;
    HYPRE_StructSMGCreate(driver->getPG()->getComm(), &precond_solver);
    HYPRE_StructSMGSetMemoryUse(precond_solver, 0);
    HYPRE_StructSMGSetMaxIter(precond_solver, 1);
    HYPRE_StructSMGSetTol(precond_solver, 0.0);
    HYPRE_StructSMGSetZeroGuess(precond_solver);
    HYPRE_StructSMGSetNumPreRelax(precond_solver, params->nPre);
    HYPRE_StructSMGSetNumPostRelax(precond_solver, params->nPost);
    HYPRE_StructSMGSetLogging(precond_solver, 0);
    _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
    _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
    _precond_solver = (HYPRE_Solver) precond_solver;
  }
}

HyprePrecondSMG::~HyprePrecondSMG(void)
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSMGDestroy((HYPRE_StructSolver) _precond_solver);
  }
}

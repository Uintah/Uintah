//--------------------------------------------------------------------------
// File: HyprePrecondSparseMSG.cc
// 
// Hypre SparseMSG (geometric multigrid #2) preconditioner.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSparseMSG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverBase.h>
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
HyprePrecondSparseMSG::initPriority(void)
  //___________________________________________________________________
  // Function HyprePrecondSparseMSG::initPriority~
  // Set the Hypre interfaces that SparseMSG can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HyprePrecondSparseMSG::setup(void)
  //___________________________________________________________________
  // Function HyprePrecondSparseMSG::setup~
  // Set up the preconditioner object. After this function call, a
  // Hypre solver can use the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreSolverParams* params = driver->getParams();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSolver precond_solver;
    HYPRE_StructSparseMSGCreate(driver->getPG()->getComm(), &precond_solver);
    HYPRE_StructSparseMSGSetMaxIter(precond_solver, 1);
    HYPRE_StructSparseMSGSetJump(precond_solver, params->jump);
    HYPRE_StructSparseMSGSetTol(precond_solver, 0.0);
    HYPRE_StructSparseMSGSetZeroGuess(precond_solver);
    /* weighted Jacobi = 1; red-black GS = 2 */
    HYPRE_StructSparseMSGSetRelaxType(precond_solver, 1);
    HYPRE_StructSparseMSGSetNumPreRelax(precond_solver, params->nPre);
    HYPRE_StructSparseMSGSetNumPostRelax(precond_solver, params->nPost);
    HYPRE_StructSparseMSGSetLogging(precond_solver, 0);
    _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSolve;
    _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSetup;
    _precond_solver = (HYPRE_Solver) precond_solver;
  }
}

HyprePrecondSparseMSG::~HyprePrecondSparseMSG(void)
  //___________________________________________________________________
  // HyprePrecondSparseMSG destructor~
  // Destroy the Hypre objects associated with the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSparseMSGDestroy((HYPRE_StructSolver) _precond_solver);
  }
}

//--------------------------------------------------------------------------
// File: HyprePrecondJacobi.cc
// 
// Hypre Jacobi (geometric multigrid #2) preconditioner.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondJacobi.h>
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
HyprePrecondJacobi::initPriority(void)
  //___________________________________________________________________
  // Function HyprePrecondJacobi::initPriority~
  // Set the Hypre interfaces that Jacobi can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HyprePrecondJacobi::setup(void)
  //___________________________________________________________________
  // Function HyprePrecondJacobi::setup~
  // Set up the preconditioner object. After this function call, a
  // Hypre solver can use the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  //  const HypreSolverParams* params = driver->getParams();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSolver precond_solver;
    HYPRE_StructJacobiCreate(driver->getPG()->getComm(), &precond_solver);
    // Number of Jacobi sweeps to be performed
    HYPRE_StructJacobiSetMaxIter(precond_solver, 2);
    HYPRE_StructJacobiSetTol(precond_solver, 0.0);
    HYPRE_StructJacobiSetZeroGuess(precond_solver);
    _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
    _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
    _precond_solver = (HYPRE_Solver) precond_solver;
  }
}

HyprePrecondJacobi::~HyprePrecondJacobi(void)
  //___________________________________________________________________
  // HyprePrecondJacobi destructor~
  // Destroy the Hypre objects associated with the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructJacobiDestroy((HYPRE_StructSolver) _precond_solver);
  }
}

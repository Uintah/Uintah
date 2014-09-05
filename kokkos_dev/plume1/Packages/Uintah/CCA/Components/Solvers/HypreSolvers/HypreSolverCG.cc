//--------------------------------------------------------------------------
// File: HypreSolverCG.cc
// 
// Hypre CG ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverCG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HypreSolverCG::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverCG::initPriority~
  // Set the Hypre interfaces that CG can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HypreSolverCG::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondCG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  const HypreSolverParams* params = _driver->getParams();
  if (_driver->getInterface() == HypreStruct) {
    HYPRE_StructSolver solver;
    HYPRE_StructPCGCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, params->maxIterations);
    HYPRE_PCGSetTol( (HYPRE_Solver)solver, params->tolerance);
    HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
    HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
    HYPRE_PCGSetLogging( (HYPRE_Solver)solver, params->logging);
    // Set up the preconditioner if we're using one
    if (_precond) {
      HYPRE_PCGSetPrecond((HYPRE_Solver)solver,
                          _precond->getPrecond(),
                          _precond->getPCSetup(),
                          HYPRE_Solver(_precond->getPrecondSolver()));
    }
    HypreDriverStruct* structDriver =
      dynamic_cast<HypreDriverStruct*>(_driver);
    // This HYPRE setup can and should be broken in the future into
    // setup that depends on HA only, and setup that depends on HB, HX.
    HYPRE_StructPCGSetup(solver,
                         structDriver->getA(),
                         structDriver->getB(),
                         structDriver->getX());
    HYPRE_StructPCGSolve(solver,
                         structDriver->getA(),
                         structDriver->getB(),
                         structDriver->getX());
    HYPRE_StructPCGGetNumIterations
      (solver, &_results.numIterations);
    HYPRE_StructPCGGetFinalRelativeResidualNorm
      (solver, &_results.finalResNorm);

    HYPRE_StructPCGDestroy(solver);
  }
}

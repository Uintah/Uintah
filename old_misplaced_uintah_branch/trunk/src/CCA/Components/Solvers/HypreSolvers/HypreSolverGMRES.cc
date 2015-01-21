//--------------------------------------------------------------------------
// File: HypreSolverGMRES.cc
// 
// Hypre GMRES ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <CCA/Components/Solvers/HypreSolvers/HypreSolverGMRES.h>
#include <CCA/Components/Solvers/HypreDriverStruct.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <SCIRun/Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HypreSolverGMRES::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverGMRES::initPriority~
  // Set the Hypre interfaces that GMRES can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HypreSolverGMRES::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondGMRES::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  const HypreSolverParams* params = _driver->getParams();
  if (_driver->getInterface() == HypreStruct) {
    HYPRE_StructSolver solver;
    HYPRE_StructGMRESCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, params->maxIterations);
    HYPRE_GMRESSetTol( (HYPRE_Solver)solver, params->tolerance );
    HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
    HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, params->logging);
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
    HYPRE_StructGMRESSetup(solver,
                         structDriver->getA(),
                         structDriver->getB(),
                         structDriver->getX());
    HYPRE_StructGMRESSolve(solver,
                         structDriver->getA(),
                         structDriver->getB(),
                         structDriver->getX());
    HYPRE_StructGMRESGetNumIterations
      (solver, &_results.numIterations);
    HYPRE_StructGMRESGetFinalRelativeResidualNorm
      (solver, &_results.finalResNorm);

    HYPRE_StructGMRESDestroy(solver);
  }
}

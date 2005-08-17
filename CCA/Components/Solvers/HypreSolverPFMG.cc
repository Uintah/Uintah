#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverPFMG.h>
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
HypreSolverPFMG::initPriority(void)
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HypreSolverPFMG::solve(void)
{
  const HypreSolverParams* params = _driver->getParams();
  if (_driver->getInterface() == HypreStruct) {
    HYPRE_StructSolver solver;
    HYPRE_StructPFMGCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_StructPFMGSetMaxIter(solver, params->maxIterations);
    HYPRE_StructPFMGSetTol(solver, params->tolerance);
    HYPRE_StructPFMGSetRelChange(solver, 0);
    /* weighted Jacobi = 1; red-black GS = 2 */
    HYPRE_StructPFMGSetRelaxType(solver, 1);
    HYPRE_StructPFMGSetNumPreRelax(solver, params->nPre);
    HYPRE_StructPFMGSetNumPostRelax(solver, params->nPost);
    HYPRE_StructPFMGSetSkipRelax(solver, params->skip);
    HYPRE_StructPFMGSetLogging(solver, params->logging);
    HypreDriverStruct* structDriver =
      dynamic_cast<HypreDriverStruct*>(_driver);
    // This HYPRE setup can and should be broken in the future into
    // setup that depends on HA only, and setup that depends on HB, HX.
    HYPRE_StructPFMGSetup(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructPFMGSolve(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructPFMGGetNumIterations
      (solver, &_results.numIterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm
      (solver, &_results.finalResNorm);

    HYPRE_StructPFMGDestroy(solver);
  }
}

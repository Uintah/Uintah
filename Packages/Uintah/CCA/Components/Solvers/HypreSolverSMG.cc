//--------------------------------------------------------------------------
// File: HypreSolverSMG.cc
// 
// Hypre SMG (geometric multigrid #1) solver.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverSMG.h>
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
HypreSolverSMG::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverSMG::initPriority~
  // Set the Hypre interfaces that SMG can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HypreSolverSMG::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondSMG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  const HypreSolverParams* params = _driver->getParams();
  if (_driver->getInterface() == HypreStruct) {
    HYPRE_StructSolver solver;
    HYPRE_StructSMGCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, params->maxIterations);
    HYPRE_StructSMGSetTol(solver, params->tolerance);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, params->nPre);
    HYPRE_StructSMGSetNumPostRelax(solver, params->nPost);
    HYPRE_StructSMGSetLogging(solver, params->logging);
    HypreDriverStruct* structDriver =
      dynamic_cast<HypreDriverStruct*>(_driver);
    // This HYPRE setup can and should be broken in the future into
    // setup that depends on HA only, and setup that depends on HB, HX.
    HYPRE_StructSMGSetup(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructSMGSolve(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructSMGGetNumIterations
      (solver, &_results.numIterations);
    HYPRE_StructSMGGetFinalRelativeResidualNorm
      (solver, &_results.finalResNorm);

    HYPRE_StructSMGDestroy(solver);
  }
}

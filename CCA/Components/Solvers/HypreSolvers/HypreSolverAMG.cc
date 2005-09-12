//--------------------------------------------------------------------------
// File: HypreSolverAMG.cc
// 
// Hypre CG ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <sci_defs/hypre_defs.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverAMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static DebugStream cout_dbg("HYPRE_DBG", false);

Priorities
HypreSolverAMG::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverAMG::initPriority~
  // Set the Hypre interfaces that AMG can work with. It can work
  // with the ParCSR interface only, anything else should be converted
  // to Par.
  // The vector of interfaces is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreParCSR);
  return priority;
}

void
HypreSolverAMG::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondCG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  cout_doing << "HypreSolverAMG::solve() BEGIN" << "\n";
  //  const HypreSolverParams* params = _driver->getParams();

  if (_driver->getInterface() == HypreSStruct) {
    // AMG parameters setup and setup phase
    HYPRE_Solver parSolver;
    HYPRE_BoomerAMGCreate(&parSolver);
    HYPRE_BoomerAMGSetCoarsenType(parSolver, 6);
    HYPRE_BoomerAMGSetStrongThreshold(parSolver, 0.);
    HYPRE_BoomerAMGSetTruncFactor(parSolver, 0.3);
    //HYPRE_BoomerAMGSetMaxLevels(parSolver, 4);
    HYPRE_BoomerAMGSetTol(parSolver, 1.0e-06);
    HYPRE_BoomerAMGSetPrintLevel(parSolver, 0);

#if HAVE_HYPRE_1_9
    HYPRE_BoomerAMGSetPrintFileName(parSolver, "sstruct.out.log");
#else
    ostringstream msg;
    cout_dbg << "Warning: Hypre version does not fully support AMG printouts "
         << "to files, proceeding without them" << "\n";
#endif // #if HAVE_HYPRE_1_9

    HYPRE_BoomerAMGSetMaxIter(parSolver, 200);
    HYPRE_BoomerAMGSetup(parSolver, _driver->getAPar(), _driver->getBPar(),
                         _driver->getXPar());
    
    // call AMG solver
    HYPRE_BoomerAMGSolve(parSolver, _driver->getAPar(), _driver->getBPar(),
                         _driver->getXPar());
  
    // Retrieve convergence information
    HYPRE_BoomerAMGGetNumIterations(parSolver,&_results.numIterations);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(parSolver,
                                                &_results.finalResNorm);
    cout_doing << "AMG convergence statistics:" << "\n";
    cout_doing << "numIterations = " << _results.numIterations << "\n";
    cout_doing << "finalResNorm  = " << _results.finalResNorm << "\n";

    // Destroy & free
    HYPRE_BoomerAMGDestroy(parSolver);
  } // interface == HypreSStruct

  cout_doing << "HypreSolverAMG::solve() END" << "\n";
}

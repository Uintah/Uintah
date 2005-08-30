//--------------------------------------------------------------------------
// File: HyprePrecondDiagonal.cc
// 
// Hypre Diagonal (geometric multigrid #2) preconditioner.
//--------------------------------------------------------------------------

#include <Packages/Uintah/CCA/Components/Solvers/HyprePreconds/HyprePrecondDiagonal.h>
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
HyprePrecondDiagonal::initPriority(void)
  //___________________________________________________________________
  // Function HyprePrecondDiagonal::initPriority~
  // Set the Hypre interfaces that Diagonal can work with. Currently, only
  // the Struct interface is supported here. The vector of interfaces
  // is sorted by descending priority.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreStruct);
  return priority;
}

void
HyprePrecondDiagonal::setup(void)
  //___________________________________________________________________
  // Function HyprePrecondDiagonal::setup~
  // Set up the preconditioner object. After this function call, a
  // Hypre solver can use the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  //  const HypreSolverParams* params = driver->getParams();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
#ifdef HYPRE_USE_PTHREADS
    for (i = 0; i < hypre_NumThreads; i++)
      precond[i] = NULL;
#else
    _precond = NULL;
#endif
    _precond = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScale;
    _pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScaleSetup;
  }
}

HyprePrecondDiagonal::~HyprePrecondDiagonal(void)
  //___________________________________________________________________
  // HyprePrecondDiagonal destructor~
  // Destroy the Hypre objects associated with the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
  }
}

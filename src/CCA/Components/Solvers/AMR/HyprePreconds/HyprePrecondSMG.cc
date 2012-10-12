/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//--------------------------------------------------------------------------
// File: HyprePrecondSMG.cc
// 
// Hypre SMG (geometric multigrid #1) preconditioner.
//--------------------------------------------------------------------------

#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondSMG.h>
#include <CCA/Components/Solvers/AMR/HypreDriver.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Solvers/HypreSolverParams.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HyprePrecondSMG::initPriority(void)
  //___________________________________________________________________
  // Function HyprePrecondSMG::initPriority~
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
HyprePrecondSMG::setup(void)
  //___________________________________________________________________
  // Function HyprePrecondSMG::setup~
  // Set up the preconditioner object. After this function call, a
  // Hypre solver can use the preconditioner.
  //___________________________________________________________________
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
  //___________________________________________________________________
  // HyprePrecondSMG destructor~
  // Destroy the Hypre objects associated with the preconditioner.
  //___________________________________________________________________
{
  const HypreDriver* driver = _solver->getDriver();
  const HypreInterface& interface = driver->getInterface();
  if (interface == HypreStruct) {
    HYPRE_StructSMGDestroy((HYPRE_StructSolver) _precond_solver);
  }
}

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
// File: HypreSolverSparseMSG.cc
// 
// Hypre SparseMSG (geometric multigrid #1) solver.
//--------------------------------------------------------------------------

#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverSparseMSG.h>
#include <CCA//Components/Solvers/AMR/HypreDriverStruct.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

Priorities
HypreSolverSparseMSG::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverSparseMSG::initPriority~
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
HypreSolverSparseMSG::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondSparseMSG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  const HypreSolverParams* params = _driver->getParams();
  if (_driver->getInterface() == HypreStruct) {
    HYPRE_StructSolver solver;
    HYPRE_StructSparseMSGCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_StructSparseMSGSetMaxIter(solver, params->maxIterations);
    HYPRE_StructSparseMSGSetJump(solver, params->jump);
    HYPRE_StructSparseMSGSetTol(solver, params->tolerance);
    HYPRE_StructSparseMSGSetRelChange(solver, 0);
    /* weighted Jacobi = 1; red-black GS = 2 */
    HYPRE_StructSparseMSGSetRelaxType(solver, 1);
    HYPRE_StructSparseMSGSetNumPreRelax(solver, params->nPre);
    HYPRE_StructSparseMSGSetNumPostRelax(solver, params->nPost);
    HYPRE_StructSparseMSGSetLogging(solver, params->logging);
    HypreDriverStruct* structDriver =
      dynamic_cast<HypreDriverStruct*>(_driver);
    // This HYPRE setup can and should be broken in the future into
    // setup that depends on HA only, and setup that depends on HB, HX.
    HYPRE_StructSparseMSGSetup(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructSparseMSGSolve(solver,
                          structDriver->getA(),
                          structDriver->getB(),
                          structDriver->getX());
    HYPRE_StructSparseMSGGetNumIterations
      (solver, &_results.numIterations);
    HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
      (solver, &_results.finalResNorm);

    HYPRE_StructSparseMSGDestroy(solver);
  }
}

/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//--------------------------------------------------------------------------
// File: HypreSolverGMRES.cc
// 
// Hypre GMRES ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverGMRES.h>
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

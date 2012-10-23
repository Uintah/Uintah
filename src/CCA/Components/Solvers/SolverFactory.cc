/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <sci_defs/hypre_defs.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/Solvers/CGSolver.h>
#include <CCA/Components/Solvers/DirectSolve.h>

#ifdef HAVE_HYPRE
#include <CCA/Components/Solvers/HypreSolver.h>
#endif

#include <CCA/Components/Solvers/AMR/AMRSolver.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace Uintah;

SolverInterface* SolverFactory::create(ProblemSpecP& ps,
                                       const ProcessorGroup* world,
                                       string cmdline)
{
  string solver = "CGSolver";

  if( cmdline == "" ) {
    ProblemSpecP sol_ps = ps->findBlock( "Solver" );
    if( sol_ps ) {
      sol_ps->getAttribute( "type", solver );
    }
  }
  else {
    solver = cmdline;
  }

  SolverInterface* solve = 0;

  if( solver == "CGSolver" ) {
    solve = scinew CGSolver(world);
  }
  else if (solver == "direct" || solver == "DirectSolver") {
    solve = scinew DirectSolve(world);
  }
  else if (solver == "HypreSolver" || solver == "hypre") {
#if HAVE_HYPRE
    solve = scinew HypreSolver2(world);
#else
    ostringstream msg;
    msg << "Hypre solver not available, hypre not configured\n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
#endif
  }
  else if (solver == "AMRSolver" || solver == "hypreamr") {
#if HAVE_HYPRE
    solve = scinew AMRSolver(world);
#else
    ostringstream msg;
    msg << "Hypre 1.9.0b solver not available, hypre not configured\n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
#endif
  }
  else {
    ostringstream msg;
    msg << "\nERROR: Unknown solver (" << solver
        << ") Valid Solvers: CGSolver, DirectSolver, HypreSolver, AMRSolver, hypreamr \n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }

  return solve;
}

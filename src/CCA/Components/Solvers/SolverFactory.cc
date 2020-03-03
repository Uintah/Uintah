/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Solvers/AMR/AMRSolver.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/Solvers/CGSolver.h>

#ifdef HAVE_HYPRE
#  include <CCA/Components/Solvers/HypreSolver.h>
#endif

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <iostream>

using namespace Uintah;

SolverInterface* SolverFactory::create(       ProblemSpecP   & ps,
                                        const ProcessorGroup * world,
                                              std::string      solverName )
{
  if( solverName == "" ) {
    ProblemSpecP sol_ps = ps->findBlock( "Solver" );
    if( sol_ps ) {
      sol_ps->getAttribute( "type", solverName );
    }
    else
      solverName = "CGSolver";
  }

  proc0cout << "Linear Solver: \t\t" << solverName << std::endl;

  SolverInterface* solver = nullptr;

  if( solverName == "CGSolver" ) {
    solver = scinew CGSolver(world);
  }
  else if (solverName == "HypreSolver" || solverName == "hypre") {
#if HAVE_HYPRE
    solver = scinew HypreSolver2(world);
#else
    std::ostringstream msg;
    msg << "\nERROR<Solver>: Hypre solver not available, Hypre was not configured.\n";
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
#endif
  }
  else if (solverName == "AMRSolver" || solverName == "hypreamr") {
#if HAVE_HYPRE
    solver = scinew AMRSolver(world);
#else
    std::ostringstream msg;
    msg << "\nERROR<Solver>: Hypre 1.9.0b solver not available, Hypre not configured.\n";
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
#endif
  }
  else {
    std::ostringstream msg;
    msg << "\nERROR<Solver>: Unknown solver (" << solverName
        << ") Valid Solvers: CGSolver, HypreSolver, AMRSolver, hypreamr \n";
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  return solver;
}

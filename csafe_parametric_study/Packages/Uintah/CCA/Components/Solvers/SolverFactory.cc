#include <sci_defs/hypre_defs.h>
#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/DirectSolve.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/AMRSolver.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace Uintah;

SolverInterface* SolverFactory::create(ProblemSpecP& ps,
                                       const ProcessorGroup* world,
                                       string cmdline)
{
  string solver = "CGSolver";;
  if (cmdline == "") {
    ProblemSpecP sol_ps = ps->findBlock("Solver");
    if (sol_ps)
      sol_ps->get("type",solver);
  }
  else
    solver = cmdline;

  SolverInterface* solve = 0;
  if(solver == "CGSolver") {
    solve = scinew CGSolver(world);
  } else {
    ostringstream msg;
    msg << "\nERROR: Unknown solver (" << solver
        << ") Valid Solvers: CGSolver, DirectSolver, HypreSolver, AMRSolver, hypreamr \n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }

  return solve;
}

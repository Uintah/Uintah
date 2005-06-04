#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/DirectSolve.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolver.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>


#include <iostream>
using std::cerr;
using std::endl;

using namespace Uintah;

SolverInterface* SolverFactory::create(ProblemSpecP& ps, 
                                       const ProcessorGroup* world)
{
  string solver = "CGSolver";

  SolverInterface* solve = 0;
  
  ps->get("SolverInterface",solver);

  if(solver == "CGSolver") {
    solve = new CGSolver(world);
  } else if(solver == "DirectSolve") {
    solve = new DirectSolve(world);
  } else if(solver == "HypreSolver" || solver == "hypre"){
#if HAVE_HYPRE
    solve = new HypreSolver2(world);
#else
    cerr << "Hypre solver not available, hypre not configured\n";
    exit(1);
#endif
  }
  
  return solve;

}

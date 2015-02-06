/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//--------------------------------------------------------------------------
// File: HypreSolverAMG.cc
// 
// Hypre CG ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <sci_defs/hypre_defs.h>
#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverAMG.h>
#include <CCA//Components/Solvers/AMR/HypreDriverStruct.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
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
  const HypreSolverParams* params = _driver->getParams();

  if (_driver->getInterface() == HypreSStruct) {
    // AMG parameters setup and setup phase
    HYPRE_Solver parSolver;
    HYPRE_BoomerAMGCreate(&parSolver);
    HYPRE_BoomerAMGSetCoarsenType(parSolver, 6);
    HYPRE_BoomerAMGSetStrongThreshold(parSolver, 0.);
    HYPRE_BoomerAMGSetTruncFactor(parSolver, 0.3);
    //HYPRE_BoomerAMGSetMaxLevels(parSolver, 4);
    HYPRE_BoomerAMGSetTol(parSolver, params->tolerance);
    HYPRE_BoomerAMGSetPrintLevel(parSolver, params->logging);
    HYPRE_BoomerAMGSetPrintFileName(parSolver, "sstruct.out.log");


    HYPRE_BoomerAMGSetMaxIter(parSolver, params->maxIterations);
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

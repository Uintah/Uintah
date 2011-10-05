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
// File: HypreSolverFAC.cc
// 
// Hypre CG ([preconditioned] conjugate gradient) solver.
//--------------------------------------------------------------------------

#include <sci_defs/hypre_defs.h>
#include <CCA/Components/Solvers/AMR/HypreSolvers/HypreSolverFAC.h>
#include <CCA/Components/Solvers/AMR/HypreDriverSStruct.h>
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
HypreSolverFAC::initPriority(void)
  //___________________________________________________________________
  // Function HypreSolverFAC::initPriority~
  // Set the Hypre interfaces that FAC can work with. Only SStruct
  // is supported.
  //___________________________________________________________________
{
  Priorities priority;
  priority.push_back(HypreSStruct);
  return priority;
}

void
HypreSolverFAC::solve(void)
  //___________________________________________________________________
  // Function HyprePrecondCG::solve~
  // Set up phase, solution stage, and destruction of all Hypre solver
  // objects.
  //___________________________________________________________________
{
  cout_doing << "HypreSolverFAC::solve() BEGIN" << "\n";
  const int numDims = 3; // Hard-coded for Uintah
  const HypreSolverParams* params = _driver->getParams();

  if (_driver->getInterface() == HypreSStruct) {
    HYPRE_SStructSolver solver;
    HypreDriverSStruct* sstructDriver =
      dynamic_cast<HypreDriverSStruct*>(_driver);
    const PatchSubset* patches = sstructDriver->getPatches();
    if (patches->size() < 1) {
      cout_dbg << "Warning: empty list of patches for FAC solver" << "\n";
      return;
    }
    const GridP grid = patches->get(0)->getLevel()->getGrid();
    int numLevels   = grid->numLevels();

    // Set the special arrays required by FAC
    int* pLevel;                  // Part ID of each level
    hypre_Index* refinementRatio; // Refinement ratio of level to level-1.
    refinementRatio = hypre_TAlloc(hypre_Index, numLevels);
    pLevel          = hypre_TAlloc(int , numLevels);
    HYPRE_SStructMatrix facA;
     for (int level = 0; level < numLevels; level++) {
      pLevel[level] = level;      // part ID of this level
      if (level == 0) {           // Dummy value
        for (int d = 0; d < numDims; d++) {
          refinementRatio[level][d] = 1;
        }
      } else {
        for (int d = 0; d < numDims; d++) {
          refinementRatio[level][d] =
            grid->getLevel(level)->getRefinementRatio()[d];
        }
      }
    }

    // Solver setup phase:
    // Prepare FAC operator hierarchy using Galerkin coarsening
    // with Dandy-black-box interpolation, on the original meshes
    hypre_AMR_RAP(sstructDriver->getA(), refinementRatio, &facA);
    // FAC parameters
    int n_pre  = refinementRatio[numLevels-1][0]-1; // # pre-relaxation sweeps
    int n_post = refinementRatio[numLevels-1][0]-1; // #post-relaxation sweeps
    // n_pre+= n_post;
    // n_post= 0;
    HYPRE_SStructFACCreate(_driver->getPG()->getComm(), &solver);
    HYPRE_SStructFACSetMaxLevels(solver, numLevels);
    HYPRE_SStructFACSetMaxIter(solver, params->maxIterations);
    HYPRE_SStructFACSetTol(solver, params->tolerance);
    HYPRE_SStructFACSetPLevels(solver, numLevels, pLevel);
    HYPRE_SStructFACSetPRefinements(solver, numLevels, refinementRatio);
    HYPRE_SStructFACSetRelChange(solver, 0);
    HYPRE_SStructFACSetRelaxType(solver, 2); // or 1
    HYPRE_SStructFACSetNumPreRelax(solver, n_pre);
    HYPRE_SStructFACSetNumPostRelax(solver, n_post);
    HYPRE_SStructFACSetCoarseSolverType(solver, 2);
    HYPRE_SStructFACSetLogging(solver, params->logging);
    HYPRE_SStructFACSetup2(solver, facA, sstructDriver->getB(),
                           sstructDriver->getX());
                           
    string warn="ERROR:\n HypreSolverFAC.cc \n  Incompatiblity with hypre 2.0.";
    throw ProblemSetupException(warn, __FILE__, __LINE__);
#if 0
    // This call isn't supported in hypre 2.0
    hypre_FacZeroCData(solver, facA, sstructDriver->getB(),
                       sstructDriver->getX());
#endif
    // Call the FAC solver
    HYPRE_SStructFACSolve3(solver, facA, sstructDriver->getB(),
                           sstructDriver->getX());

    // Retrieve convergence information
    HYPRE_SStructFACGetNumIterations(solver, &_results.numIterations);
    HYPRE_SStructFACGetFinalRelativeResidualNorm(solver,
                                                 &_results.finalResNorm);
    cout_dbg << "FAC convergence statistics:" << "\n";
    cout_dbg << "numIterations = " << _results.numIterations << "\n";
    cout_dbg << "finalResNorm  = " << _results.finalResNorm << "\n";

    // Destroy & free
    HYPRE_SStructFACDestroy2(solver);
    hypre_TFree(pLevel);
    hypre_TFree(refinementRatio);
    HYPRE_SStructGraph facGraph = hypre_SStructMatrixGraph(facA);
    HYPRE_SStructGraphDestroy(facGraph);
    HYPRE_SStructMatrixDestroy(facA);

  } // interface == HypreSStruct

  cout_doing << "HypreSolverFAC::solve() END" << "\n";
}

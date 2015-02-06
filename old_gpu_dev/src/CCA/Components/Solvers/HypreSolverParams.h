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


#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h

/*--------------------------------------------------------------------------
CLASS
   HypreSolverParams
   
   Parameters struct for AMRSolver and HypreDriver.

GENERAL INFORMATION

   File: HypreSolverParams.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   AMRSolver, HypreDriver, HypreSolverParams, HypreSolverBase.

DESCRIPTION
   Parameters struct for AMRSolver and HypreDriver.
 
WARNING

   --------------------------------------------------------------------------*/

#include <CCA/Ports/SolverInterface.h>

namespace Uintah {

  class HypreSolverParams : public SolverParameters {

  public:

    HypreSolverParams(void) 
      {
        printSystem = true;
        timing      = true;
      }

    ~HypreSolverParams(void) {}

    // Parameters common for all Hypre Solvers
    string solverTitle;        // String corresponding to solver type
    string precondTitle;       // String corresponding to preconditioner type
    double tolerance;          // Residual tolerance for solver
    int    maxIterations;      // Maximum # iterations allowed
    int    logging;            // Log Hypre solver (using Hypre options)
    bool   symmetric;          // Is LHS matrix symmetric
    bool   restart;            // Allow solver to restart if not converged

    // SMG parameters
    int    nPre;               // # pre relaxations for Hypre SMG solver
    int    nPost;              // # post relaxations for Hypre SMG solver

    // PFMG parameters
    int    skip;               // Hypre PFMG parameter

    // SparseMSG parameters
    int    jump;               // Hypre Sparse MSG parameter

    // Debugging and control flags
    bool   printSystem;    // Linear system dump to file
    bool   timing;         // Time results

  }; 
}

#endif

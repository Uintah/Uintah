/*--------------------------------------------------------------------------
CLASS
   HypreSolverParams
   
   Parameters struct for HypreSolverAMR and HypreInterface.

GENERAL INFORMATION

   File: HypreSolverParams.h

   Oren E. Livne
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   HypreSolverAMR, HypreInterface, HypreSolverParams, HypreGenericSolver.

DESCRIPTION
   Parameters struct for HypreSolverAMR and HypreInterface.
 
WARNING

   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h

//#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

namespace Uintah {

  class HypreSolverParams : public SolverParameters {

    /*========================== PUBLIC SECTION ==========================*/
  public:

    HypreSolverParams() {}
    ~HypreSolverParams() {}

    /* Parameters common for all Hypre Solvers */
    string solverTitle;      // String corresponding to solver type
    string precondTitle;     // String corresponding to preconditioner type
    double tolerance;        // Residual tolerance for solver
    int maxIterations;       // Maximum # iterations allowed
    int logging;             // Log Hypre solver (using Hypre options)
    bool symmetric;          // Is LHS matrix symmetric
    bool restart;            // Allow solver to restart if not converged
    //    SolverType solverType;   // Hypre Solver type
    //    PrecondType precondType; // Hypre Preconditioner type

    /* TODO: Move/copy these parameters to each specific solver? */

    /* SMG parameters */
    int nPre;                // # pre relaxations for Hypre SMG solver
    int nPost;               // # post relaxations for Hypre SMG solver

    /* PFMG parameters */
    int skip;                // Hypre PFMG parameter

    /* SparseMSG parameters */
    int jump;                // Hypre Sparse MSG parameter

    /*===== Oren's extra parameters, might be removed later. =====*/
    /* Debugging and control flags */
    bool            printSystem;    // Linear system dump to file
    bool            timing;         // Time results
    bool            saveResults;    // Dump the solution, error to files
    int             verboseLevel;   // Verbosity level of debug printouts

  }; // class HypreSolverParams
} // end namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h

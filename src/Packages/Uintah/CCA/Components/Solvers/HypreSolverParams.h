/*--------------------------------------------------------------------------
 * File: HypreSolverParams.h
 *
 * Parameters struct for HypreSolverAMR and HypreSolverWrap.
 * and corresponding solvers. When adding a new Hypre solver, remember to
 * update this class and the functions involving solverType in HypreSolverAMR
 * and HypreSolverWrap.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverParams_h

//#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

namespace Uintah {

  class HypreSolverParams : public SolverParameters {
    /*_____________________________________________________________________
      class HypreSolverParams:
      Input parameters struct.
      Add to this struct all parameters of all Hypre solvers to be used
      so that we can control them from the same section of the sus input
      file.
      _____________________________________________________________________*/
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
    SolverType solverType;   // Hypre Solver type
    PrecondType precondType; // Hypre Preconditioner type

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

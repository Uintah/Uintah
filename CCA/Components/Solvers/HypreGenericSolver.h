/*--------------------------------------------------------------------------
 * File: HypreGenericSolver.h
 *
 * class HypreGenericSolver is a generic Hypre solver that takes data from
 * HypreDriver according to what the specific solver can work with, and outputs
 * results back into the appropriate data structures in HypreDriver.
 * Preconditioners are also a type of solver.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h
#define Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>

class HypreGenericSolver {
  /*_____________________________________________________________________
    class HypreGenericSolver:
    A base (generic) solver handler that gets all the necessary data
    pointers (A,b,x,...), solves the linear system by calling some Hypre
    solver (implemented in derived classes from HypreGenericSolver),
    and returns some output statistics and the solution vector.
    _____________________________________________________________________*/

  /*========================== PUBLIC SECTION ==========================*/
 public:
  
    /*---------- Types ----------*/

    enum SolverType {
      SMG, PFMG, SparseMSG, CG, Hybrid, GMRES, AMG, FAC
    };

    enum PrecondType {
      PrecondNA, // No preconditioner, use solver directly
      PrecondSMG, PrecondPFMG, PrecondSparseMSG, PrecondJacobi,
      PrecondDiagonal, PrecondAMG, PrecondFAC
    };

  /* Solver results are output to this struct */
  struct Results {
    Counter    numIterations;   // Number of solver iterations performed
    double     finalResNorm;    // Final residual norm ||A*x-b||_2
  };

  virtual ~HypreGenericSolver(void) {
    Print("Destroying HypreGenericSolver object\n");
    
    /* Destroy graph objects */
    Print("Destroying graph objects\n");
    HYPRE_SStructGraphDestroy(_graph);
    
    /* Destroy matrix, RHS, solution objects */
    Print("Destroying matrix, RHS, solution objects\n");
    HYPRE_SStructMatrixDestroy(_A);
    HYPRE_SStructVectorDestroy(_b);
    HYPRE_SStructVectorDestroy(_x);
   
  }

  void initialize(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructStencil& stencil);

  virtual void setup(void);
  virtual void solve(void);

  /* Utilities */
  virtual void printMatrix(const string& fileName = "output");
  virtual void printRHS(const string& fileName = "output_b");
  virtual void printSolution(const string& fileName = "output_x");

  /*----- Data Members -----*/
  const Param*          _param;
  bool                  _requiresPar;   // Does solver require Par input?
  Counter               _solverID;      // Hypre solver ID
  Results               _results;       // HypreGenericSolver results are outputted to here

  /*========================== PROTECTED SECTION ==========================*/
 protected:

  HypreGenericSolver(const Param* param)
    : _param(param)
    {
      _results.numIterations = 0;
      _results.finalResNorm  = DBL_MAX;
    }

  virtual void initializeData(const Hierarchy& hier,
                              const HYPRE_SStructGrid& grid);
  virtual void assemble(void);
  
  void printValues(const Patch* patch,
                   const int stencilSize,
                   const int numCells,
                   const double* values = 0,
                   const double* rhsValues = 0,
                   const double* solutionValues = 0);

  /* Utilities */

  static SolverType solverFromTitle(const string& solverTitle);
  static PrecondType precondFromTitle(const string& precondTitle);
  static HypreDriver::Interface solverInterface(const SolverType& solverType);

  /*========================== PRIVATE SECTION ==========================*/
 private:

  /*===== Data Members =====*/
  const Param*          _param;
  bool                  _requiresPar;   // Does solver require Par input?
  Counter               _solverID;      // Hypre solver ID
  Results               _results;       // HypreGenericSolver results are outputted to here

  void makeGraph(const Hierarchy& hier,
                 const HYPRE_SStructGrid& grid,
                 const HYPRE_SStructStencil& stencil);
  void makeLinearSystem(const Hierarchy& hier,
                        const HYPRE_SStructGrid& grid,
                        const HYPRE_SStructStencil& stencil);

};

#endif // Packages_Uintah_CCA_Components_Solvers_HypreGenericSolver_h

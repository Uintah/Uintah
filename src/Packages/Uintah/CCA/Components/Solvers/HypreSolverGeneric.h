/*--------------------------------------------------------------------------
 * File: HypreSolverGeneric.h
 *
 * class HypreSolverGeneric is a generic Hypre solver that takes data from
 * HypreDriver according to what the specific solver can work with, and outputs
 * results back into the appropriate data structures in HypreDriver.
 * Preconditioners are also a type of solver.
 *--------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_HypreSolverGeneric_h
#define Packages_Uintah_CCA_Components_Solvers_HypreSolverGeneric_h

#include <Packages/Uintah/CCA/Components/Solvers/HypreDrive.h>

class HypreSolverGeneric {
  /*_____________________________________________________________________
    class HypreSolverGeneric:
    A base (generic) solver handler that gets all the necessary data
    pointers (A,b,x,...), solves the linear system by calling some Hypre
    solver (implemented in derived classes from HypreSolverGeneric),
    and returns some output statistics and the solution vector.
    _____________________________________________________________________*/

  /*========================== PUBLIC SECTION ==========================*/
 public:
  
  /* Solver results are output to this struct */
  struct Results {
    Counter    numIterations;   // Number of solver iterations performed
    double     finalResNorm;    // Final residual norm ||A*x-b||_2
  };

  virtual ~HypreSolverGeneric(void) {
    Print("Destroying HypreSolverGeneric object\n");
    
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
  Results               _results;       // HypreSolverGeneric results are outputted to here

  /*========================== PROTECTED SECTION ==========================*/
 protected:

  HypreSolverGeneric(const Param* param)
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

  /*========================== PRIVATE SECTION ==========================*/
 private:

  /*===== Data Members =====*/
  const Param*          _param;
  bool                  _requiresPar;   // Does solver require Par input?
  Counter               _solverID;      // Hypre solver ID
  Results               _results;       // HypreSolverGeneric results are outputted to here

  void makeGraph(const Hierarchy& hier,
                 const HYPRE_SStructGrid& grid,
                 const HYPRE_SStructStencil& stencil);
  void makeLinearSystem(const Hierarchy& hier,
                        const HYPRE_SStructGrid& grid,
                        const HYPRE_SStructStencil& stencil);

};

#endif // Packages_Uintah_CCA_Components_Solvers_HypreSolverGeneric_h

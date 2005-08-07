#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "Hierarchy.h"
#include "util.h"

// Hypre libraries
#include <HYPRE_sstruct_ls.h>
#include <utilities.h>
#include <krylov.h>
#include <sstruct_mv.h>
#include <sstruct_ls.h>

class Solver {
  /*_____________________________________________________________________
    class Solver:
    A base (generic) solver handler that gets all the necessary data
    pointers (A,b,x,...), solves the linear system by calling some Hypre
    solver (implemented in derived classes from Solver),
    and returns some output statistics and the solution vector.
    _____________________________________________________________________*/

  /*========================== PUBLIC SECTION ==========================*/
 public:
  
  struct Results {
    Counter    numIterations;   // Number of solver iterations performed
    double     finalResNorm;    // Final residual norm ||A*x-b||_2
  };

  virtual ~Solver(void) {
    dbg << "Destroying Solver object" << "\n";
    
    /* Destroy graph objects */
    dbg << "Destroying graph objects" << "\n";
    HYPRE_SStructGraphDestroy(_graph);
    
    /* Destroy matrix, RHS, solution objects */
    dbg << "Destroying matrix, RHS, solution objects" << "\n";
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
  Results               _results;       // Solver results are outputted to here

  /*========================== PROTECTED SECTION ==========================*/
 protected:

  Solver(const Param* param)
    : _param(param)
    {
      _results.numIterations = 0;
      _results.finalResNorm  = 1234.5678; //DBL_MAX;
    }

  virtual void initializeData(const Hierarchy& hier,
                              const HYPRE_SStructGrid& grid);
  virtual void assemble(void);
  
  /*===== Data Members =====*/

  /* SStruct objects */ // We assume Solver is an SStruct solver
  HYPRE_SStructMatrix   _A;
  HYPRE_SStructVector   _b;
  HYPRE_SStructVector   _x;
  HYPRE_SStructGraph    _graph; // Graph needed for all SStruct Solvers

  /* ParCSR objects */
  HYPRE_ParCSRMatrix    _parA;
  HYPRE_ParVector       _parB;
  HYPRE_ParVector       _parX;

  /*========================== PRIVATE SECTION ==========================*/
 private:
  void makeGraph(const Hierarchy& hier,
                 const HYPRE_SStructGrid& grid,
                 const HYPRE_SStructStencil& stencil);

  void makeUnderlyingIdentity(const Counter level,
			      const HYPRE_SStructStencil& stencil,
			      const Box& coarseUnderFine);
  void makeLinearSystem(const Hierarchy& hier,
                        const HYPRE_SStructGrid& grid,
                        const HYPRE_SStructStencil& stencil);

  void printValues(const Counter numCells,
                   const Counter stencilSize,
                   const double* values = 0,
                   const double* rhsValues = 0,
                   const double* solutionValues = 0);
};

#endif // __SOLVER_H__

#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "mydriver.h"
#include "Param.h"
#include "Hierarchy.h"
#include "util.h"
#include <values.h>
#include <vector>
#include <string>
#include <map>
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
 public:
  
  struct Results {
    Counter    numIterations;   // Number of solver iterations performed
    double     finalResNorm;    // Final residual norm ||A*x-b||_2
  };

  Solver(const Param* param)
    : _param(param)
    {
      _results.numIterations = 0;
      _results.finalResNorm  = DBL_MAX;
    }

  virtual ~Solver(void) {
    Print("Destroying Solver object\n");
    
    /* Destroy matrix, RHS, solution objects */
    Print("Destroying matrix, RHS, solution objects\n");
    HYPRE_SStructMatrixDestroy(_A);
    HYPRE_SStructVectorDestroy(_b);
    HYPRE_SStructVectorDestroy(_x);
   
  }

  void initialize(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructStencil& stencil,
                  const HYPRE_SStructGraph& graph);

  virtual void setup(void) = 0;
  virtual void solve(void) = 0;

  /* Utilities */
  virtual void printMatrix(const string& fileName = "output");
  virtual void printRHS(const string& fileName = "output_b");
  virtual void printSolution(const string& fileName = "output_x");

  /*======================= Data Members =============================*/
  const Param*          _param;
  Counter               _solverID;      // Hypre solver ID
  bool                  _requiresPar;   // Does solver require Par input?

  /* SStruct objects */
  HYPRE_SStructMatrix   _A;
  HYPRE_SStructVector   _b;
  HYPRE_SStructVector   _x;

  /* ParCSR objects */
  HYPRE_ParCSRMatrix    _parA;
  HYPRE_ParVector       _parB;
  HYPRE_ParVector       _parX;

  /* Solver results */
  Results               _results;   // Solver results are outputted to here

 protected:
  virtual void initializeData(const Hierarchy& hier,
                              const HYPRE_SStructGrid& grid,
                              const HYPRE_SStructGraph& graph);
  void makeLinearSystem(const Hierarchy& hier,
                        const HYPRE_SStructGrid& grid,
                        const HYPRE_SStructStencil& stencil);
  virtual void assemble(void);

 private:
};

#endif // __SOLVER_H__

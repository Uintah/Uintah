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
    A solver handler that gets all the necessary data pointers (A,b,x,...),
    solves the linear system by calling Hypre, and returns some output 
    statistics and the solution vector.
    _____________________________________________________________________*/
public:
  
  struct Results {
    Counter    numIterations;   // Number of solver iterations performed
    double     finalResNorm;    // Final residual norm ||A*x-b||_2
  };

  /* Data for solver */
  const Param&          _param;
  Counter               _solverID;  // Type of Hypre solver

  /* SStruct objects */
  HYPRE_SStructMatrix   _A;
  HYPRE_SStructVector   _b;
  HYPRE_SStructVector   _x;

  /* ParCSR objects */
  HYPRE_ParCSRMatrix    _parA;
  HYPRE_ParVector       _parB;
  HYPRE_ParVector       _parX;

  /* FAC objects */
  HYPRE_SStructMatrix   _facA;
  int*                  _pLevel;          // Needed by FAC: part # of level
  Index*                _refinementRatio; // Needed by FAC

  Results               _results;   // Solver results are outputted to here


  Solver(const Param& param)
    : _param(param)
    {
      _results.numIterations = 0;
      _results.finalResNorm  = DBL_MAX;
    }

  ~Solver(void) {
    Print("Destroying Solver object\n");
    hypre_TFree(_pLevel);
    hypre_TFree(_refinementRatio);

    Print("Destroying graph objects\n");
    if (_param.solverID > 90) {
      HYPRE_SStructGraph facGraph = hypre_SStructMatrixGraph(_facA);
      HYPRE_SStructGraphDestroy(facGraph);
    }
    
    /* Destroy matrix, RHS, solution objects */
    Print("Destroying matrix, RHS, solution objects\n");
    if (_param.solverID > 90) {
      HYPRE_SStructMatrixDestroy(_facA);
    }
    HYPRE_SStructMatrixDestroy(_A);
    HYPRE_SStructVectorDestroy(_b);
    HYPRE_SStructVectorDestroy(_x);
   
  }

  void initialize(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructGraph& graph);
  void assemble(void);
  void setup(void);
private:
};

#endif // __SOLVER_H__

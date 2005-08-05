#ifndef __SOLVERFAC_H__
#define __SOLVERFAC_H__

#include "Solver.h"

class SolverFAC : public Solver {
  /*_____________________________________________________________________
    class SolverFAC:
    A solver handler that gets all the necessary data pointers (A,b,x,...),
    solves the linear system by calling Hypre, and returns some output 
    statistics and the solution vector.
    _____________________________________________________________________*/
public:
  
  SolverFAC(const Param* param)
    : Solver(param)
    {
      _solverID = 99;
    }

  ~SolverFAC(void) {
    Print("Destroying SolverFAC object\n");
    hypre_TFree(_pLevel);
    hypre_TFree(_refinementRatio);

    Print("Destroying graph objects\n");
    HYPRE_SStructGraph facGraph = hypre_SStructMatrixGraph(_facA);
    HYPRE_SStructGraphDestroy(facGraph);
    
    /* Destroy matrix, RHS, solution objects */
    Print("Destroying matrix, RHS, solution objects\n");
    HYPRE_SStructMatrixDestroy(_facA);
  }

  virtual void setup(void);
  virtual void solve(void);
  
  /* Utilities */
  virtual void printMatrix(const string& fileName = "output");

private:
  void initializeData(const Hierarchy& hier,
                      const HYPRE_SStructGrid& grid);

  //  void assemble(void);

  /* FAC objects */
  HYPRE_SStructMatrix   _facA;
  int*                  _pLevel;          // Needed by FAC: part # of level
  hypre_Index*          _refinementRatio; // Needed by FAC
};

#endif // __SOLVERFAC_H__

#ifndef __SOLVERAMG_H__
#define __SOLVERAMG_H__

#include "Solver.h"

class SolverAMG : public Solver {
  /*_____________________________________________________________________
    class SolverAMG:
    Solve the linear system with BoomerAMG (Hypre solver ID = 30).
    _____________________________________________________________________*/
public:
  
  SolverAMG(const Param* param)
    : Solver(param)
    {
      _solverID = 30;
    }

  virtual ~SolverAMG(void) {
    Print("Destroying SolverAMG object\n");
  }

  void solve(void);

private:
  //  virtual void assemble(void);
  void setup(void);

  HYPRE_Solver  _parSolver;
};

#endif // __SOLVERAMG_H__

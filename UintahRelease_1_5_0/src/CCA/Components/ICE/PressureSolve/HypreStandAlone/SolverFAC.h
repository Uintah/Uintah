/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
    dbg << "Destroying SolverFAC object" << "\n";
    hypre_TFree(_pLevel);
    hypre_TFree(_refinementRatio);

    dbg << "Destroying graph objects" << "\n";
    HYPRE_SStructGraph facGraph = hypre_SStructMatrixGraph(_facA);
    HYPRE_SStructGraphDestroy(facGraph);
    
    /* Destroy matrix, RHS, solution objects */
    dbg << "Destroying matrix, RHS, solution objects" << "\n";
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

#ifndef __TESTLINEAR_H__
#define __TESTLINEAR_H__

#include "Param.h"

class TestLinear : public Param {
  /*_____________________________________________________________________
    class TestLinear:
    Test case with
    * Exact solution U = linear function with Dirichlet B.C. on the d-D
    unit square. U is smooth.
    * Diffusion a=1 (Laplace operator).
    _____________________________________________________________________*/
public:
  
  TestLinear(void) : Param()
    /* Constructor: initialize this test case's default parameters */
    {
      /* Problem parameters */
      numDims  = 2;        // # dimensions
      longTitle = "TestLinear";      // Title of this test case

      /* log files, output types */
      outputDir  = longTitle;      // Directory of output files
      logFile    = longTitle + ".log";        // File logging run flow

      /* Domain geometry & coarsest grid */

      /* Debugging and control flags */
    }
  
  virtual ~TestLinear(void) {}
  

  /* Input functions */
  
  virtual double diffusion(const Location& x);     // Diffusion coefficient
  virtual double rhs(const Location& x);           // Right-hand-side of PDE
  virtual double rhsBC(const Location& x);         // RHS of B.C.
  virtual double exactSolution(const Location& x); // Exact solution

};

#endif // __TESTLINEAR_H__

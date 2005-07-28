#ifndef __TESTLINEAR_H__
#define __TESTLINEAR_H__

#include "Param.h"
#include "util.h"

class TestLinear : public Param {
  /*_____________________________________________________________________
    class TestLinear:
    Test case with
    * Exact solution U = linear function with Dirichlet B.C. on the d-D
    unit square. U is smooth.
    * Diffusion a=1 (Laplace operator).
    _____________________________________________________________________*/
public:
  
  TestLinear(const Counter d,
             const Counter base) : Param()
    /* Constructor: initialize this test case's default parameters */
    {
      /* Problem parameters */
      setNumDims(d);
#if 0
      /* Boundary conditions for a rectangular domain */
      // All Dirichlet in this case
      vector<Patch::BoundaryCondition> bc(2*numDims);
      for (int d = 0; d < numDims; d++) {
        bc[2*d  ] = Patch::Dirichlet;
        bc[2*d+1] = Patch::Dirichlet;
      }
     
      setDomain(base,bc); // Set rectangular domain with BC = bc
#endif
      baseResolution = base;

      /* log files, output types */
      longTitle = "TestLinear";      // Title of this test case
      outputDir  = longTitle;      // Directory of output files
      logFile    = longTitle + ".log";        // File logging run flow

      /* Grid hierarchy */
      numLevels = 2;
      twoLevelType = CentralHalf;
      threeLevelType = CentralHalf;

      /* Debugging and control flags */
    }
  
  virtual ~TestLinear(void) {}
  

  /* Input functions */
  
  virtual double diffusion(const Location& x) const;     // Diffusion coefficient
  virtual double rhs(const Location& x) const;           // Right-hand-side of PDE
  virtual double rhsBC(const Location& x) const;         // RHS of B.C.
  virtual double exactSolution(const Location& x) const; // Exact solution

};

#endif // __TESTLINEAR_H__

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
             const Counter base);
  virtual ~TestLinear(void) {}
  
  /* Input functions */
  
  virtual double diffusion(const Location& x) const;     // Diffusion coeff
  virtual double rhs(const Location& x) const;           // RHS of PDE
  virtual double rhsBC(const Location& x) const;         // RHS of B.C.
  virtual double exactSolution(const Location& x) const; // Exact solution

};

#endif // __TESTLINEAR_H__

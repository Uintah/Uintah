#include "TestLinear.h"

using namespace std;

double TestLinear::diffusion(const Location& x) const
  // Diffusion coefficient
{
  return 1.0;
}

double TestLinear::rhs(const Location& x) const
  // Right-hand-side of PDE
{
  return 0.0;
}

double TestLinear::rhsBC(const Location& x) const
  // RHS of B.C.
  // Dirichlet B.C. at all boundaries
{
  return exactSolution(x);
}

double TestLinear::exactSolution(const Location& x) const
  // Exact solution
  // U is a linear function (d-D)
{
  double u = 1.0;
  for (Counter d = 0; d < numDims; d++) {
    u += x[d];
  }
  return u;
}

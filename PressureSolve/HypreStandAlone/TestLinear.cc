#include "TestLinear.h"

using namespace std;

double TestLinear::diffusion(const Location& x)
  // Diffusion coefficient
{
  return 1.0;
}

double TestLinear::rhs(const Location& x)
  // Right-hand-side of PDE
{
  return 0.0;
}

double TestLinear::rhsBC(const Location& x)
  // RHS of B.C.
  // Dirichlet B.C. at all boundaries
{
  return exactSolution(x);
}

double TestLinear::exactSolution(const Location& x)
  // Exact solution
  // U is a linear function (d-D)
{
  double u = 1.0;
  for (int d = 0; d < numDims; d++) {
    u += x[d];
  }
  return u;
}

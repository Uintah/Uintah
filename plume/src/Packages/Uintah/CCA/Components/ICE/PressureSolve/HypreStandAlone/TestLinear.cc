#include "TestLinear.h"

using namespace std;

TestLinear::TestLinear(const Counter d,
                       const Counter base) :
  Param()
  // Constructor: initialize this test case's default parameters */
{
  /* Problem parameters */
  setNumDims(d);
#if 0
  /* Boundary conditions for a rectangular domain */
  // All Dirichlet in this case
  Vector<Patch::BoundaryCondition> bc(2*numDims);
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

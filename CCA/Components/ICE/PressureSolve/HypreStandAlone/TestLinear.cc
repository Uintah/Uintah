/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
  //  return 1.0;
  //  return random()/RAND_MAX;

  bool upperLeft = true;
  for (Counter d = 0; d < numDims; d++) {
    if (abs(x[d]) <= 0.5) {
      upperLeft = false;
      break;
    }
  }
  if (upperLeft) return 1e+10;
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

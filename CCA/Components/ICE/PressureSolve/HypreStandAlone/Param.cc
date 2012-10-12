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

#include "Param.h"
#include "util.h"

using namespace std;

void Param::setNumDims(const Counter d)
{
  if (numDims) {
    cerr << "\n\nError: numDims already initialized and cannot be re-set."
         << "\n";
    clean();
    exit(1);
  }
  numDims = d;
}

#if 0
void Param::setDomain(const Counter baseResolution,
                      const vector<Patch::BoundaryCondition>& bc)
  /* Domain of physical size [0,1]x[0,1] with coarsest meshsize h
     = 1/baseResolution. */
{
  if (domain) delete domain;
  domain = scinew Level(numDims,1.0/baseResolution);
  vector<int> lower(numDims,0), upper(numDims,baseResolution);
  Patch* box = scinew Patch(-1,-1,Box(lower,upper));
  box->setAllBC(bc);
  domain->_patchList.push_back(box);
}
#endif

double Param::harmonicAvg(const Location& x,
                          const Location& y,
                          const Location& z) const
  /*_____________________________________________________________________
    Function makeGrid: 
    Harmonic average of the diffusion coefficient.
    A = harmonicAvg(X,Y,Z) returns the harmonic average of the
    diffusion coefficient a(T) (T in R^D) along the line connecting
    the points X,Y in R^D. That is, A = 1/(integral_0^1
    1/a(t1(s),...,tD(s)) ds), where td(s) = x{d} + s*(y{d} -
    x{d})/norm(y-x) is the arclength parameterization of the
    d-coordinate of the line x-y, d = 1...D.  We assume that A is
    piecewise constant with jump at Z (X,Y are normally cell centers
    and Z at the cell face). X,Y,Z are Dx1 location arrays.  In
    general, A can be analytically computed for the specific cases we
    consider; in general, use some simple quadrature formula for A
    from discrete a-values. This can be implemented by the derived
    test cases from Param.

    ### NOTE: ### If we use a different
    refinement ratio in different dimensions, near the interface we
    may need to compute A along lines X-Y that cross more than one
    cell boundary. This is currently ignored and we assume all lines
    cut one cell interface only.
    _____________________________________________________________________*/

{
  double Ax = diffusion(x);
  double Ay = diffusion(y);
  /* Compute distances x-y and x-z */
  double dxy = 0.0, dxz = 0.0;
  for (Counter d = 0; d < numDims; d++) {
    dxy += pow(fabs(y[d] - x[d]),2.0);
    dxz += pow(fabs(z[d] - x[d]),2.0);
  }
  double K = sqrt(dxz/dxy);
  return (Ax*Ay)/((1-K)*Ax + K*Ay);
}

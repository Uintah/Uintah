#include "Param.h"

using namespace std;

double Param::harmonicAvg(const Location& x,
                          const Location& y,
                          const Location& z)
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
    and Z at the cell face). X,Y,Z are Dx1 cell arrays, we treat every
    element of X{d},Y{d},Z{d} separately and output A as an array of
    size size(X{1}).  In general, A can be analytically computedfor
    the specific cases we consider; in general, use some simple
    quadrature formula for A from discrete a-values. This can be
    implemented by the derived test cases from Param.

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

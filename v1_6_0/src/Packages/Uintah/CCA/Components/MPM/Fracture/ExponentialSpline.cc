#include "ExponentialSpline.h"

#include <Core/Geometry/Vector.h>

namespace Uintah {
double
ExponentialSpline::
w(const Vector& r) const
{
  double s = r.length() / radius;
  return ws(s);
}

double
ExponentialSpline::
dwdx(const int i,const Vector& r) const
{
  double r0 = r.length();
  double s = r0 / radius;
  return dwsds(s) * r(i) / r0;
}

double
ExponentialSpline::
ws(const double s) const
{
  double t = s / radius;
  if(s <= 1) return exp(-t*t);
  else return 0;
}

double
ExponentialSpline::
dwsds(const double s) const
{
  double t = s / _alpha;
  if(s <= 1) return 2 * s * t * t * exp(-t*t);
  else return 0;
}

void
ExponentialSpline::
setAlpha(double alpha)
{
  _alpha = alpha;
}
} // End namespace Uintah



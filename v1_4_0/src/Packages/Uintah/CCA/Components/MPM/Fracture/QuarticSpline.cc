#include "QuarticSpline.h"

#include <Core/Geometry/Vector.h>

namespace Uintah {
double
QuarticSpline::
w(const Vector& r) const
{
  double s = r.length() / radius;
  return ws(s);
}

double
QuarticSpline::
dwdx(const int i,const Vector& r) const
{
  double r0 = r.length();
  double s = r0 / radius;
  return dwsds(s) * r(i) / r0;
}

double
QuarticSpline::
ws(const double s) const
{
  if(s <= 1) return 1.-6.*s*s+8.*s*s*s-3.*s*s*s*s;
  else return 0;
}

double
QuarticSpline::
dwsds(const double s) const
{
  if(s <= 1) return -12.*s+24.*s*s-12.*s*s*s;
  else return 0;
}
} // End namespace Uintah



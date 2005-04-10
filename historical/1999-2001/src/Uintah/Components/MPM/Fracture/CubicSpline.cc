#include "CubicSpline.h"

#include <SCICore/Geometry/Vector.h>

namespace Uintah {
namespace MPM {

double
CubicSpline::
w(const Vector& r) const
{
  double s = r.length() / radius;
  return ws(s);
}

double
CubicSpline::
dwdx(const int i,const Vector& r) const
{
  double r0 = r.length();
  double s = r0 / radius;
  return dwsds(s) * r(i) / r0;
}

double
CubicSpline::
ws(const double s) const
{
  if(s <= 0.5) return 2./3. - 4.*s*s + 4.*s*s*s;
  else if(s <= 1) return 4./3. - 4.*s + 4.*s*s - 4./3.*s*s*s;
  else return 0;
}

double
CubicSpline::
dwsds(const double s) const
{
  if(s <= 0.5) return - 8.*s + 12.*s*s;
  else if(s <= 1) return - 4. + 8.*s - 4.*s*s;
  else return 0;
}

}} //namespace

// $Log$
// Revision 1.1  2000/07/06 00:06:48  tan
// Added CubicSpline class for 3D least square approximation.
//

#include "CrackFace.h"

namespace Uintah {

void CrackFace::setup(const Vector& n,const Point& p)
{
}

double CrackFace::distance(const Point& p)
{
  double D = _a*_a + _b*_b +_c*_c;
  ASSERT(D>0);
  
  return ( _a*p.x()+_b*p.y()+_c*p.z()+_d ) / sqrt(D);
}


} //namespace

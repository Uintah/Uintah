#ifndef UINTAH_MPM_LEASTSQUARE
#define UINTAH_MPM_LEASTSQUARE

#include "Equation.h"

namespace SCIRun {
  class Vector;
}

namespace Uintah {

using SCIRun::Vector;

class Spline;

class LeastSquare {
public:

  void     input(const Vector& x,double y);
  void     output(double& a, Vector& b);
  void     clean();
  
           LeastSquare(const Spline& spline);

private:
  Equation        _equ;
  const Spline*   _spline;
};

} // End namespace Uintah

#endif

#ifndef Uintah_MPM_LeastSquare
#define Uintah_MPM_LeastSquare

#include "Equation.h"

namespace SCICore {
namespace Geometry {
  class Vector;
}}

namespace Uintah {
namespace MPM {

class Spline;
using SCICore::Geometry::Vector;

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

}} //namespace

#endif

// $Log$
// Revision 1.2  2000/07/06 06:23:42  tan
// Added Least Square interpolation of double (such as temperatures),
// vector (such as velocities) and stresses for particles in the
// self-contact cells.
//
// Revision 1.1  2000/07/06 04:58:19  tan
// Added LeastSquare class.
//

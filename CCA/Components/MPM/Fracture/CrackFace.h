#ifndef Uintah_MPM_CrackFace
#define Uintah_MPM_CrackFace

#include <Core/Geometry/Vector.h>

namespace Uintah {

using SCIRun::Vector;
using SCIRun::Point;

class CrackFace {
public:
               CrackFace(double a,double b,double c,double d) :
	         _a(a),_b(b),_c(c),_d(d) {};
	       CrackFace() {};

  void         setup(const Vector& n,const Point& p);
	       
  double       distance(const Point& p);

private:
  double      _a,_b,_c,_d;
};

} //namespace

#endif

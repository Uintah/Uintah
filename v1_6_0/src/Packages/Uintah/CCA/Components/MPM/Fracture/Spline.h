#ifndef __MPM_SPLINE__
#define __MPM_SPLINE__

namespace SCIRun {
  class Vector;
}

namespace Uintah {

using namespace SCIRun;

class Spline {
public:
          double             radius;

  virtual double             w(const Vector& r) const = 0;
  virtual double             dwdx(int i,const Vector& r) const = 0;
};

} // End namespace Uintah

#endif

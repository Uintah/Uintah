#ifndef _MPM_ExponentialSpline
#define _MPM_ExponentialSpline

#include "Spline.h"

namespace Uintah {
class ExponentialSpline : public Spline {
public:

        void               setAlpha(double alpha);
  
        double             w(const Vector& r) const;
        double             dwdx(const int i,const Vector& r) const;

private:
        double             ws(const double s) const;
        double             dwsds(const double s) const;

private:
  double  _alpha;
};
} // End namespace Uintah


#endif


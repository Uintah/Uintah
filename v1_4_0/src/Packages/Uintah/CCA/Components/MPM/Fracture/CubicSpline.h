#ifndef UINTAH_MPM_CUBICSPLINE
#define UINTAH_MPM_CUBICSPLINE

#include "Spline.h"

namespace Uintah {

class CubicSpline : public Spline {
public:
        double             w(const Vector& r) const;
        double             dwdx(int i,const Vector& r) const;

private:
        double             ws(const double s) const;
        double             dwsds(const double s) const;
};
} // End namespace Uintah

#endif


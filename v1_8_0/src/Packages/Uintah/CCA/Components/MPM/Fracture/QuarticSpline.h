#ifndef _MPM_QuarticSpline
#define _MPM_QuarticSpline

#include "Spline.h"

namespace Uintah {
class QuarticSpline : public Spline {
public:

        double             w(const Vector& r) const;
        double             dwdx(const int i,const Vector& r) const;

private:
        double             ws(const double s) const;
        double             dwsds(const double s) const;

};
} // End namespace Uintah


#endif


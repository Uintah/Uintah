#ifndef Uintah_MPM_ExponentialSpline
#define Uintah_MPM_ExponentialSpline

#include "Spline.h"

namespace Uintah {
namespace MPM {

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

}} //namespace

#endif

// $Log$
// Revision 1.1  2000/07/06 00:31:47  tan
// Added ExponentialSpline class for 3D least square approximation.
//

#ifndef Uintah_MPM_CubicSpline
#define Uintah_MPM_CubicSpline

#include "Spline.h"

namespace Uintah {
namespace MPM {

class CubicSpline : public Spline {
public:
        double             w(const Vector& r) const;
        double             dwdx(int i,const Vector& r) const;

private:
        double             ws(const double s) const;
        double             dwsds(const double s) const;
};

}} //namespace

#endif

// $Log$
// Revision 1.1  2000/07/06 00:06:25  tan
// Added CubicSpline class for 3D least square approximation.
//

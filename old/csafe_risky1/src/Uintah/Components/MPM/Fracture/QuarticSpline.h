#ifndef Uintah_MPM_QuarticSpline
#define Uintah_MPM_QuarticSpline

#include "Spline.h"

namespace Uintah {
namespace MPM {

class QuarticSpline : public Spline {
public:

        double             w(const Vector& r) const;
        double             dwdx(const int i,const Vector& r) const;

private:
        double             ws(const double s) const;
        double             dwsds(const double s) const;

};

}} //namespace

#endif

// $Log$
// Revision 1.1  2000/07/06 00:23:51  tan
// Added QuarticSpline class for 3D least square approximation.
//

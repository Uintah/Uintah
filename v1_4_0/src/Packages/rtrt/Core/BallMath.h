/***** BallMath.h - Essential routines for Arcball.  *****/
#ifndef _H_BallMath
#define _H_BallMath
#include <Packages/rtrt/Core/BallAux.h>

//namespace rtrt {

rtrt::HVect MouseOnSphere(rtrt::HVect mouse, rtrt::HVect ballCenter, double ballRadius);
rtrt::HVect ConstrainToAxis(rtrt::HVect loose, rtrt::HVect axis);
int NearestConstraintAxis(rtrt::HVect loose, rtrt::HVect *axes, int nAxes);
rtrt::Quat Qt_FromBallPoints(rtrt::HVect from, rtrt::HVect to);
void Qt_ToBallPoints(rtrt::Quat q, rtrt::HVect *arcFrom, rtrt::HVect *arcTo);

//} // end namespace rtrt

#endif

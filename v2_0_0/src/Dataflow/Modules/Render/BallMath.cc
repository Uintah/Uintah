/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/**** BallMath.c - Essential routines for ArcBall.  ****/
#include <math.h>
#include "BallMath.h"
#include "BallAux.h"


/* Convert window coordinates to sphere coordinates. */
HVect MouseOnSphere(HVect mouse, HVect ballCenter, double ballRadius)
{
    HVect ballMouse;
    register double mag;
    ballMouse.x = (mouse.x - ballCenter.x) / ballRadius;
    ballMouse.y = (mouse.y - ballCenter.y) / ballRadius;
    mag = ballMouse.x*ballMouse.x + ballMouse.y*ballMouse.y;
    if (mag > 1.0) {
	register double scale = 1.0/sqrt(mag);
	ballMouse.x *= scale; ballMouse.y *= scale;
	ballMouse.z = 0.0;
    } else {
	ballMouse.z = sqrt(1 - mag);
    }
    ballMouse.w = 0.0;
    return (ballMouse);
}

/* Construct a unit quaternion from two points on unit sphere */
Quat Qt_FromBallPoints(HVect from, HVect to)
{
    Quat qu;
    qu.x = from.y*to.z - from.z*to.y;
    qu.y = from.z*to.x - from.x*to.z;
    qu.z = from.x*to.y - from.y*to.x;
    qu.w = from.x*to.x + from.y*to.y + from.z*to.z;
    return (qu);
}

/* Convert a unit quaternion to two points on unit sphere */
void Qt_ToBallPoints(Quat q, HVect *arcFrom, HVect *arcTo)
{
    double s;
    s = sqrt(q.x*q.x + q.y*q.y);
    if (s == 0.0) {
	*arcFrom = V3_(0.0, 1.0, 0.0);
    } else {
	*arcFrom = V3_(-q.y/s, q.x/s, 0.0);
    }
    arcTo->x = q.w*arcFrom->x - q.z*arcFrom->y;
    arcTo->y = q.w*arcFrom->y + q.z*arcFrom->x;
    arcTo->z = q.x*arcFrom->y - q.y*arcFrom->x;
    if (q.w < 0.0) *arcFrom = V3_(-arcFrom->x, -arcFrom->y, 0.0);
}

/* Force sphere point to be perpendicular to axis. */
HVect ConstrainToAxis(HVect loose, HVect axis)
{
    HVect onPlane;
    register double norm;
    onPlane = V3_Sub(loose, V3_Scale(axis, V3_Dot(axis, loose)));
    norm = V3_Norm(onPlane);
    if (norm > 0.0) {
	if (onPlane.z < 0.0) onPlane = V3_Negate(onPlane);
	return ( V3_Scale(onPlane, 1/sqrt(norm)) );
    } /* else drop through */
    if (axis.z == 1) {
	onPlane = V3_(1.0, 0.0, 0.0);
    } else {
	onPlane = V3_Unit(V3_(-axis.y, axis.x, 0.0));
    }
    return (onPlane);
}

/* Find the index of nearest arc of axis set. */
int NearestConstraintAxis(HVect loose, HVect *axes, int nAxes)
{
    HVect onPlane;
    register double max, dot;
    register int i, nearest;
    max = -1; nearest = 0;
    for (i=0; i<nAxes; i++) {
	onPlane = ConstrainToAxis(loose, axes[i]);
	dot = V3_Dot(onPlane, loose);
	if (dot>max) {
	    max = dot; nearest = i;
	}
    }
    return (nearest);
}

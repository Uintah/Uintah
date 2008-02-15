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

/***** BallAux.h - Vector and quaternion routines for Arcball. *****/
#ifndef _H_BallAux
#define _H_BallAux

#include <math.h>

enum QuatPart {X, Y, Z, W, QuatLen};
typedef double HMatrix[QuatLen][QuatLen];

class Quat {
public:
    Quat() {}
    Quat(double a,double b,double c, double d) {
	x = a;
	y = b;
	z = c;
	w = d;
    }

    void ToMatrix(HMatrix& out);
    void FromMatrix(HMatrix&);
    Quat operator * (Quat& q) {
	Quat ret;
	ret.w = w*q.w - x*q.x - y*q.y - z*q.z;
	ret.x = w*q.x + x*q.w + y*q.z - z*q.y;
	ret.y = w*q.y + y*q.w + z*q.x - x*q.z;
	ret.z = w*q.z + z*q.w + x*q.y - y*q.x;
	return ret;
    }	
    Quat Conj(void) { 
	Quat qq;
	qq.x = -x; qq.y = -y; qq.z = -z; qq.w = w;
	return (qq);
    }
	
    double VecMag(void) { return sqrt(x*x + y*y + z*z); };
    double x, y, z, w;
};
typedef Quat HVect;

extern Quat qOne;
HVect V3_(double x, double y, double z);
double V3_Norm(HVect v);
HVect V3_Unit(HVect v);
HVect V3_Scale(HVect v, double s);
HVect V3_Negate(HVect v);
HVect V3_Sub(HVect v1, HVect v2);
double V3_Dot(HVect v1, HVect v2);
HVect V3_Cross(HVect v1, HVect v2);
HVect V3_Bisect(HVect v0, HVect v1);
#endif

/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

/***** BallAux.h - Vector and quaternion routines for Arcball. *****/
#ifndef _H_BallAux
#define _H_BallAux

#include <math.h>

namespace rtrt {

#ifndef Bool
typedef int Bool;
#endif
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

  //extern Quat qOne;
} // end namespace rtrt

rtrt::HVect V3_(double x, double y, double z);
double V3_Norm(rtrt::HVect v);
rtrt::HVect V3_Unit(rtrt::HVect v);
rtrt::HVect V3_Scale(rtrt::HVect v, double s);
rtrt::HVect V3_Negate(rtrt::HVect v);
rtrt::HVect V3_Sub(rtrt::HVect v1, rtrt::HVect v2);
double V3_Dot(rtrt::HVect v1, rtrt::HVect v2);
rtrt::HVect V3_Cross(rtrt::HVect v1, rtrt::HVect v2);
rtrt::HVect V3_Bisect(rtrt::HVect v0, rtrt::HVect v1);

#endif


#ifndef Math_Spline_h
#define Math_Spline_h 1

namespace rtrt {

inline double CRSpline(double t, double p1, double p2,
		double p3, double p4)
{
	double t1=((-1*t+2)*t-1)*t;
	double t2=((3*t-5)*t+0)*t+2;
	double t3=((-3*t+4)*t+1)*t;
	double t4=((1*t-1)*t+0)*t;
	return (p1*t1+p2*t2+p3*t3+p4*t4)*0.5;
}

inline double dCRSpline(double t, double p1, double p2,
	double p3, double p4)
{
	double t1=(-3*t+4)*t-1;
	double t2=(9*t-10)*t;
	double t3=(-9*t+8)*t+1;
	double t4=(3*t-2)*t;
	return (p1*t1+p2*t2+p3*t3+p4*t4)*0.5;
}

} // end namespace rtrt

#endif

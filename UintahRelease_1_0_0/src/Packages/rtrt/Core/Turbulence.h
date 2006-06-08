
#ifndef Math_Turbulence_h
#define Math_Turbulence_h 1

#include <Packages/rtrt/Core/Noise.h>

namespace SCIRun {
  class Point;
}

namespace rtrt {

using SCIRun::Point;

class Turbulence {
	Noise noise;
	int noctaves;
	double s;
	double a;
public:
	Turbulence(int=4,double=0.5,double=2.0,int=0,int=4096);
	Turbulence(const Turbulence&);
	double operator()(const Point&);
	double operator()(const Point&, double);
	Vector dturb(const Point&, double);
	Vector dturb(const Point&, double, double&);
};

} // end namespace rtrt

#endif

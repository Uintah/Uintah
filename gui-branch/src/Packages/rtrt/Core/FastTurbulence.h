
#ifndef Math_FastTurbulence_h
#define Math_FastTurbulence_h 1

#include "FastNoise.h"

namespace rtrt {

class Point;

class FastTurbulence {
	FastNoise noise;
	int noctaves;
	double s;
	double a;
public:
	FastTurbulence(int=6,double=0.5,double=2.0,int=0,int=4096);
	FastTurbulence(const FastTurbulence&);
	double operator()(const Point&);
	double operator()(const Point&, double);
	Vector dturb(const Point&, double);
	Vector dturb(const Point&, double, double&);
};

} // end namespace rtrt

#endif

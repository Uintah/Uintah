#ifndef SOLIDNOISE3_H
#define SOLIDNOISE3_H


#include <stdlib.h>
#include <math.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Dot;

class SolidNoise3 {
private:
    void Permute(int* a, int n)
    {
	long i, j;
	int temp;
    
	for (i = n-1; i > 0; i--) {
	    j = lrand48() % (i+1);
	    temp = a[i];
	    a[i] = a[j];
	    a[j] = temp;
	}
    }

public:
    Vector grad[256];
    int phi[256];
    SolidNoise3();
    double noise(const Point&) const;
    Vector vectorNoise(const Point&) const;
    Vector vectorTurbulence( const Point&, int) const ;
    double turbulence(const Point&, int) const;
    double dturbulence(const Point&, int, double) const;
    double omega(double) const;
    Vector gamma(int, int, int) const;
    int intGamma(int, int) const;
    double knot(int, int, int, Vector&) const;
    Vector vectorKnot(int, int, int, Vector&) const;
};


inline double SolidNoise3::omega(double t) const {
   t = (t > 0.0)? t : -t;
   return (t < 1.0)?  ((2*t - 3)*t*t  + 1) : 0.0;
}

inline Vector SolidNoise3::gamma(int i, int j, int k) const
{
   int idx;
   idx = phi[abs(k)%256];
   idx = phi[abs(j + idx) % 256];
   idx = phi[abs(i + idx) % 256];
   return grad[idx];
}

inline double SolidNoise3::knot(int i, int j, int k, Vector& v) const {
  return ( omega(v.x()) * omega(v.y()) * omega(v.z()) * Dot(gamma(i,j,k),v));
}

inline Vector SolidNoise3::vectorKnot( int i, int j, int k, Vector& v)
const {
    return ( omega(v.x()) * omega(v.y()) * omega(v.z()) * gamma(i,j,k) );
}

inline int SolidNoise3::intGamma(int i, int j) const {
   int idx;
   idx = phi[abs(j)%256];
   idx = phi[abs(i + idx) % 256];
   return idx;
}

} // end namespace rtrt

#endif

/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef SOLIDNOISE3_H
#define SOLIDNOISE3_H


#include <cstdlib>
#include <cmath>
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

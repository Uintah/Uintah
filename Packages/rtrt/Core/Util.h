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


#include <cmath>
#include <cstdio>

#include <Core/Geometry/Vector.h>

#ifndef UTIL_H
#define UTIL_H

//namespace rtrt {

void jacobi(double **, int, double [], double **, int *);
int solvecubic(double [4], double [3]);

#define EQN_EPS 1E-12

inline int IsZero(double x)
{
    return x > -EQN_EPS && x < EQN_EPS;
}

inline int NotZero(double x)
{
    return x < -EQN_EPS || x > EQN_EPS;
}

#define NFACT 13

extern unsigned int fact_table[NFACT];

inline unsigned int fact(int n)
{
    unsigned int f=1;

    if ((n >= 0) && (n < NFACT))
	return fact_table[n];
    if (n<0)
	return 1;
    f = fact_table[NFACT-1];
    for (int i=NFACT; i<=n; i++) {
	f *= i;
    }
    
    return f;
}

#if 0
inline double pow(double x, int y)
{
    double res=1;
    double m;

    if (y < 0) {
	x = 1./x;
	y = -y;
    }
    m = x;
    while(y) {
	if ( y & 1 )
	    res *= m;
	y = y >> 1;
	m *= m;
    }
    return res;
}
#endif

#define NCOMB 35

extern unsigned int comb_table[NCOMB][NCOMB];
extern int comb_table_inited;

inline void init_comb_table()
{
    int n, i;

    if (comb_table_inited)
      return;
    comb_table_inited = 1;

    comb_table[0][0] = 1;
    
    for (n=1; n<NCOMB; n++) {
	comb_table[n][0] = 1;
	for (i=1; i<n; i++) 
	    comb_table[n][i] = comb_table[n-1][i-1]+comb_table[n-1][i];
	comb_table[n][n] = 1;
    }
}

//} // end namespace rtrt

#endif

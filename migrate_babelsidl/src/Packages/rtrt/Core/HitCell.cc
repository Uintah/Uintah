#include    <Core/Geometry/Point.h>
#include    <Core/Geometry/Vector.h>
#include    <Packages/rtrt/Core/Ray.h>
#include <iostream>

using namespace rtrt;
using namespace std;

int SolveQuadratic(double *c, double *s);
int SolveCubic( double c[ 4 ], double s[ 3 ]);

namespace rtrt {
int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
            float rho[2][2][2], float iso, double tmin, double tmax, double& t) {
    double c[ 4 ];
    double s[ 3 ];
    double ua[2];
    double ub[2];
    double va[2];
    double vb[2];
    double wa[2];
    double wb[2];
    int i,j,k;

#ifdef DEBUG_P
    
    cerr << "pmin=" << pmin << ", pmax=" << pmax << '\n';
    cerr << "tmin=" << tmin << ", tmax=" << tmax << "\n";
#endif
    ua[1] = (r.origin().x() - pmin.x()) / (pmax.x() - pmin.x());
    ua[0] = 1 - ua[1];
    ub[1] = r.direction().x() / (pmax.x() - pmin.x());
    ub[0] = - ub[1];

    va[1] = (r.origin().y() - pmin.y()) / (pmax.y() - pmin.y());
    va[0] = 1 - va[1];
    vb[1] = r.direction().y() / (pmax.y() - pmin.y());
    vb[0] = - vb[1];

    wa[1] = (r.origin().z() - pmin.z()) / (pmax.z() - pmin.z());
    wa[0] = 1 - wa[1];
    wb[1] = r.direction().z()  / (pmax.z() - pmin.z());
    wb[0] = - wb[1];


    c[3] = c[2] = c[1] = c[0] = 0;
    for (i=0; i < 2; i++)
       for (j=0; j < 2; j++)
         for (k=0; k < 2; k++) {
#ifdef DEBUG_P
	   cerr << "rho[" << i << "][" << j << "][" << k << "]=" << rho[i][j][k] << '\n';
#endif
            // cubic term
            c[3] += ub[i]*vb[j]*wb[k] * rho[i][j][k]; 

            // square term
            c[2] += (ua[i]*vb[j]*wb[k] + ub[i]*va[j]*wb[k] + ub[i]*vb[j]*wa[k]) * rho[i][j][k]; 

            // linear term
            c[1] += (ub[i]*va[j]*wa[k] + ua[i]*vb[j]*wa[k] + ua[i]*va[j]*wb[k]) * rho[i][j][k]; 

            // constant term
            c[0] +=  ua[i]*va[j]*wa[k] * rho[i][j][k]; 
         }
    c[0] -= iso;
#ifdef DEBUG_P
    cerr << "c: " << c[3] << ", " << c[2] << ", " << c[1] << ", " << c[0] << '\n';
#endif

    int n = SolveCubic( c,  s);


    t = tmax;
    for (i = 0; i < n; i++) {
#ifdef DEBUG_P
      cerr << "s[" << i << "]: " << s[i] << '\n';
#endif
       if (s[i] >= tmin && s[i] < t) {
            t = s[i];
       }
    }

    return (t < tmax);
}

int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
            float rho[2][2], double tmin, double tmax, double& t)
{
    double c[ 3 ];
    double s[ 2 ];
    double ua[2];
    double ub[2];
    double va[2];
    double vb[2];
    int i,j;

#ifdef DEBUG_P
    
    cerr << "pmin=" << pmin << ", pmax=" << pmax << '\n';
    cerr << "tmin=" << tmin << ", tmax=" << tmax << "\n";
#endif
    ua[1] = (r.origin().x() - pmin.x()) / (pmax.x() - pmin.x());
    ua[0] = 1 - ua[1];
    ub[1] = r.direction().x() / (pmax.x() - pmin.x());
    ub[0] = - ub[1];

    va[1] = (r.origin().y() - pmin.y()) / (pmax.y() - pmin.y());
    va[0] = 1 - va[1];
    vb[1] = r.direction().y() / (pmax.y() - pmin.y());
    vb[0] = - vb[1];

    c[2] = c[1] = c[0] = 0;
    for (i=0; i < 2; i++){
       for (j=0; j < 2; j++){
#ifdef DEBUG_P
	  cerr << "rho[" << i << "][" << j << "]=" << rho[i][j] << '\n';
#endif
	  // square term
	  c[2] += (ub[i]*vb[j]) * rho[i][j]; 

	  // linear term
	  c[1] += (ub[i]*va[j] + ua[i]*vb[j]) * rho[i][j]; 

	  // constant term
	  c[0] +=  ua[i]*va[j] * rho[i][j]; 
       }
    }
    c[0] -= r.origin().z();
    c[1] -= r.direction().z();
#ifdef DEBUG_P
    cerr << "c: " << c[2] << ", " << c[1] << ", " << c[0] << '\n';
#endif

    int n = SolveQuadratic( c,  s);


    t = tmax;
    for (i = 0; i < n; i++) {
#ifdef DEBUG_P
      cerr << "s[" << i << "]: " << s[i] << '\n';
#endif
       if (s[i] >= tmin && s[i] < t) {
            t = s[i];
       }
    }

    return (t < tmax);
}

} // end namespace rtrt


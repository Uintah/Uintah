
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;  

void make_ortho(const Vector& v, Vector& v1, Vector& v2) 
{
    Vector v0(Cross(v, Vector(1,0,0)));
    if(v0.length2() == 0){
        v0=Cross(v, Cross(v, Vector(0,1,0)));
    }
    v1=Cross(v, v0);
    v1.normalize();
    v2=Cross(v, v1);
    v2.normalize();
}

// returns n^2 unit vectors within a cone with interior angle 2*theta0
// oriented in direction axis (need not be unit).  Assumes incone is
// already allocated
void get_cone_vectors(const Vector& axis, double theta0, int n, Vector* incone)
{
     Vector what = axis;
     what.normalize();
     Vector uhat, vhat;
     make_ortho(what, uhat, vhat);
     double cosTheta0 = cos(theta0);
     int k = 0;
     for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
           double phi = 2 * M_PI * (i + drand48()) / n;
           double cosTheta = 1 - (1 - cosTheta0) * (j + drand48()) / n;
           double sinTheta = sqrt(1 - cosTheta*cosTheta);
           incone[k++] = uhat*cos(phi)*sinTheta +
                         vhat*sin(phi)*sinTheta +
                         what*cosTheta;
        }
     }
}

Light::Light(const Point& pos, const Color& color, double radius)
    : pos(pos), color(color), radius(radius)
{
    Vector d(pos-Point(0,0,.5));
    int n=3;
    beamdirs.resize(n*n);
    get_cone_vectors(d, .03, n, &beamdirs[0]);
}

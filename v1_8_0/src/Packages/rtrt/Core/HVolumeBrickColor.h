
#ifndef HVOLUMEBRICKCOLOR_H
#define HVOLUMEBRICKCOLOR_H 1

#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/Material.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

namespace rtrt {

using SCIRun::WorkQueue;

class HVolumeBrickColor : public Material {
protected:
    Point min;
    Vector datadiag;
    Vector sdiag;
    double dt;
    int nx,ny,nz;
    unsigned char* indata;
    unsigned char* blockdata;
    unsigned long* xidx;
    unsigned long* yidx;
    unsigned long* zidx;
    WorkQueue work;
    void brickit(int);
    char* filebase;
    double Ka, Kd, Ks, specpow, refl;
    bool grid;
    bool nn;
    friend class HVolumeBrickColorDpy;
public:
    HVolumeBrickColor(char* filebase, int np, double Ka, double Kd,
		      double Ks, double specpow, double refl,
		      double dt=0);
    virtual ~HVolumeBrickColor();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif

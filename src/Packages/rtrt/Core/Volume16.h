
#ifndef VOLUME16_H
#define VOLUME16_H 1

#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

class Volume16 : public VolumeBase {
protected:
    Point min;
    Vector diag;
    Vector sdiag;
    int nx,ny,nz;
    short* data;
    short datamin, datamax;
    friend class aVolume16;
public:
    Volume16(Material* matl, VolumeDpy* dpy, char* filebase);
    virtual ~Volume16();
    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_hist(int nhist, int* hist,
			      float datamin, float datamax);
    virtual void get_minmax(float& min, float& max);
};

} // end namespace rtrt

#endif

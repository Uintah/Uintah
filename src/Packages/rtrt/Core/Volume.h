
#ifndef VOLUME_H
#define VOLUME_H 1

#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>

#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

class Volume : public VolumeBase {
protected:
    Point min;
    Vector diag;
    Vector sdiag;
    int nx,ny,nz;
    float* data;
    float datamin, datamax;
    friend class aVolume;
public:
    Volume(Material* matl, VolumeDpy* dpy, char* filebase);
    virtual ~Volume();
    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_hist(int nhist, int* hist,
			      float datamin, float datamax);
    virtual void get_minmax(float& min, float& max);
};

class aVolume : public VolumeBase {
protected:
    Array1< Volume * > vols; // volumes for this timer series...
    int nx,ny,nz;

  int ctime; // current time...

    float datamin, datamax;
public:
    aVolume(Material* matl, VolumeDpy*, char* filebase);
    virtual ~aVolume();
    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_hist(int nhist, int* hist,
			      float datamin, float datamax);
    virtual void get_minmax(float& min, float& max);
    virtual void animate(double t, bool& changed);
};


} // end namespace rtrt

#endif


#ifndef VOLUMEBASE_H
#define VOLUMEBASE_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {

class VolumeDpy;

class VolumeBase : public Object {
protected:
    VolumeDpy* dpy;
public:
    VolumeBase(Material* matl, VolumeDpy* dpy);
    virtual ~VolumeBase();
    virtual void animate(double t, bool& changed);
    virtual void compute_hist(int nhist, int* hist,
			      float datamin, float datamax)=0;
    virtual void get_minmax(float& min, float& max)=0;
};

} // end namespace rtrt

#endif

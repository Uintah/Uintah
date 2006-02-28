
#ifndef RTRT_VolumeVGBASE_H
#define RTRT_VolumeVGBASE_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {
  
class Hist2DDpy;

class VolumeVGBase : public Object {
protected:
  Hist2DDpy* dpy;
public:
  VolumeVGBase(Material* matl, Hist2DDpy* dpy);
  virtual ~VolumeVGBase();
  virtual void animate(double t, bool& changed);
  virtual void compute_hist(int nxhist, int nyhist, int** hist,
			    float vdatamin, float vdatamax,
			    float gdatamin, float gdatamax)=0;
  virtual void get_minmax(float& vmin, float& vmax,
			  float& gmin, float& gmax)=0;
};

} // end namespace rtrt

#endif // #ifndef RTRT_VolumeVGBASE_H

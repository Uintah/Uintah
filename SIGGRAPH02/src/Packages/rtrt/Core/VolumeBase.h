
#ifndef VOLUMEBASE_H
#define VOLUMEBASE_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {
class VolumeBase;
class VolumeDpy;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::VolumeBase*&);
}

namespace rtrt {

class VolumeBase : public Object {
protected:
  VolumeDpy* dpy;
public:
  VolumeBase(Material* matl, VolumeDpy* dpy);
  virtual ~VolumeBase();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, VolumeBase*&);

  virtual void animate(double t, bool& changed);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax)=0;
  virtual void get_minmax(float& min, float& max)=0;
};

} // end namespace rtrt

#endif

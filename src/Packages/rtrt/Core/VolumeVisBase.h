
#ifndef VOLUMEVISBASE_H
#define VOLUMEVISBASE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>

namespace rtrt {
class VolumeVisBase;
class VolumeVisDpy;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::VolumeVisBase*&);
}

namespace rtrt {

class VolumeVisBase : public Object, public Material {
protected:
  VolumeVisDpy* dpy;
public:
  VolumeVisBase(VolumeVisDpy* dpy);
  virtual ~VolumeVisBase();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, VolumeVisBase*&);

  virtual void animate(double t, bool& changed);
  virtual void compute_hist(int nhist, int* hist,
    float datamin, float datamax){}
  virtual void get_minmax(float& min, float& max){};
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx) = 0;
};

} // end namespace rtrt

#endif


#include <Packages/rtrt/Core/RingSatellite.h>

namespace rtrt {
using namespace SCIRun;

Persistent* ring_satellite_maker() {
  return new RingSatellite();
}

// initialize the static member type_id
PersistentTypeID RingSatellite::type_id("RingSatellite", "Object", ring_satellite_maker);

void RingSatellite::animate(double /*t*/, bool& changed)
{
  cen = parent_->get_center();
  d=Dot(this->n, cen);
  changed = true;
}
  
void RingSatellite::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  // radial mapping of a 1D texture
  double hitdist = (hitpos-cen).length();
  double edge1 = hitdist-radius;
  double edge2 = thickness;
  double u = edge1/edge2;
  uv.set(u,0);
}

const int RING_SATELLITE_VERSION = 1;

void 
RingSatellite::io(SCIRun::Piostream &str)
{
  str.begin_class("RingSatellite", RING_SATELLITE_VERSION);
  Ring::io(str);
  SCIRun::Pio(str, parent_);
  str.end_class();
}
} // end namespace

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::RingSatellite*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::RingSatellite::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::RingSatellite*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun


#include <Packages/rtrt/Core/UVPlane.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* uvp_maker() {
  return new UVPlane();
}

// initialize the static member type_id
PersistentTypeID UVPlane::type_id("UVPlane", "UVMapping", uvp_maker);

UVPlane::UVPlane(const Point& cen, const Vector& v1, const Vector& v2)
    : cen(cen), v1(v1), v2(v2)
{
}

UVPlane::~UVPlane()
{
}

void UVPlane::uv(UV& uv, const Point& hitpos, const HitInfo&)
{
    Vector p(hitpos-cen);
    double uu=Dot(v1, p);
    double vv=Dot(v2, p);
    uv.set(uu,vv);
}

const int UVPLANE_VERSION = 1;

void 
UVPlane::io(SCIRun::Piostream &str)
{
  str.begin_class("UVPlane", UVPLANE_VERSION);
  UVMapping::io(str);
  Pio(str, cen);
  Pio(str, v1);
  Pio(str, v2);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::UVPlane*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::UVPlane::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::UVPlane*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun


#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Vector.h>


using namespace rtrt;
using namespace SCIRun;

// initialize the static member type_id
PersistentTypeID UVMapping::type_id("UVMapping", "Persistent", 0);

UVMapping::UVMapping()
{
}

UVMapping::~UVMapping()
{
}

void UVMapping::get_frame(const Point &, const HitInfo&,const Vector &norm, 
			  Vector &v2, Vector &v3) 
{
  Vector v(1,0,0);
  v2=Cross(norm,v);
  if (v2.length2()<1.e-8) {
    v=Vector(0,1,0);
    v2=Cross(norm,v);
  }
  v2.normalize();
  v3=Cross(norm,v2);
  v3.normalize();
}

const int UVMAPPING_VERSION = 1;

void 
UVMapping::io(SCIRun::Piostream &str)
{
  str.begin_class("UVMapping", UVMAPPING_VERSION);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::UVMapping*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::UVMapping::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::UVMapping*>(pobj);
    //ASSERT(obj != 0);
  }
}
} // end namespace SCIRun


#include <Core/Util/Assert.h>
#include <Packages/rtrt/Core/Shadows/NoShadows.h>
using namespace rtrt;
using namespace SCIRun;

Persistent* noShadows_maker() {
  return new NoShadows();
}

// initialize the static member type_id
PersistentTypeID NoShadows::type_id("NoShadows", "ShadowBase", 
				      noShadows_maker);


NoShadows::NoShadows()
{
}

NoShadows::~NoShadows()
{
}

bool NoShadows::lit(const Point& hitpos, Light* light,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  return true;
}

const int NOSHADOWS_VERSION = 1;

void 
NoShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("NoShadows", NOSHADOWS_VERSION);
  ShadowBase::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::NoShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::NoShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::NoShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

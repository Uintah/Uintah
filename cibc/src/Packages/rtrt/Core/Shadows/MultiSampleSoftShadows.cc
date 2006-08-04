
#include <Packages/rtrt/Core/Shadows/MultiSampleSoftShadows.h>
#include <Packages/rtrt/Core/Array1.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* multiSampleSoftShadows_maker() {
  return new MultiSampleSoftShadows();
}

// initialize the static member type_id
PersistentTypeID MultiSampleSoftShadows::type_id("MultiSampleSoftShadows", 
						 "ShadowBase", 
						 multiSampleSoftShadows_maker);

MultiSampleSoftShadows::MultiSampleSoftShadows()
{
}

MultiSampleSoftShadows::~MultiSampleSoftShadows()
{
}


bool MultiSampleSoftShadows::lit(const Point& hitpos, Light* light,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  // These are busted.  Do not use Array1....
#if 0
  Array1<Vector>& beamdirs=light->get_beamdirs();
  int n=beamdirs.size();
  for(int i=0;i<n;i++)
    attens[i]=Color(1,1,1);
  obj->multi_light_intersect(light, hitpos, beamdirs, attens,
			     dist, &cx->stats->ds[depth], ppc);
  atten = Color(0,0,0);
  for(int i=0;i<n;i++)
    atten+=attens[i];
  atten = atten*(1.0/n);
#endif
  return true;
}

const int MULTISAMPLESHADOWS_VERSION = 1;

void 
MultiSampleSoftShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("MultiSampleSoftShadows", MULTISAMPLESHADOWS_VERSION);
  ShadowBase::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::MultiSampleSoftShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::MultiSampleSoftShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MultiSampleSoftShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

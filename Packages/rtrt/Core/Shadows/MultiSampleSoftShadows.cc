
#include <Packages/rtrt/Core/Shadows/MultiSampleSoftShadows.h>
#include <Packages/rtrt/Core/Array1.h>
using namespace rtrt;


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


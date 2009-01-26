/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Packages/rtrt/Core/Shadows/SingleSampleSoftShadows.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
using namespace rtrt;
using namespace SCIRun;

Persistent* singleSampleShadows_maker() {
  return new SingleSampleSoftShadows();
}

// initialize the static member type_id
PersistentTypeID SingleSampleSoftShadows::type_id("SingleSampleSoftShadows", 
						  "ShadowBase", 
						  singleSampleShadows_maker);

#define MAXDEPTH 200

SingleSampleSoftShadows::SingleSampleSoftShadows()
{
}

SingleSampleSoftShadows::~SingleSampleSoftShadows()
{
}

void SingleSampleSoftShadows::preprocess(Scene*, int& pp_offset, int&)
{
  shadow_cache_offset = pp_offset;
  pp_offset += sizeof(Object*)*MAXDEPTH;
}

bool SingleSampleSoftShadows::lit(const Point& hitpos, Light* light,
				  const Vector& light_dir, double dist, Color& atten,
				  int depth, Context* cx)
{
  HitInfo hit;
  hit.min_t = dist;
  Ray lightray(hitpos, light_dir);
  Object* obj=cx->scene->get_shadow_object();
  Object** shadow_cache = (Object**)cx->ppc->get(shadow_cache_offset,
						 sizeof(Object*)*MAXDEPTH);
  if(shadow_cache[depth]){
    shadow_cache[depth]->softshadow_intersect(light, lightray, hit, dist, atten,
					      &cx->stats->ds[depth], cx->ppc);
    cx->stats->ds[depth].shadow_cache_try++;
    if(hit.was_hit){
      return false;
    }
    shadow_cache[depth]=0;
    cx->stats->ds[depth].shadow_cache_miss++;
  }
  obj->softshadow_intersect(light, lightray, hit, dist, atten,
			    &cx->stats->ds[depth], cx->ppc);
  
  if(hit.was_hit){
    shadow_cache[depth]=hit.hit_obj;
    return false;
  }
  shadow_cache[depth]=0;
  return true;
}

const int SINGLESAMPLESHADOWS_VERSION = 1;

void 
SingleSampleSoftShadows::io(SCIRun::Piostream &str)
{
  str.begin_class("SingleSampleSoftShadows", SINGLESAMPLESHADOWS_VERSION);
  ShadowBase::io(str);
  SCIRun::Pio(str, shadow_cache_offset);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::SingleSampleSoftShadows*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::SingleSampleSoftShadows::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::SingleSampleSoftShadows*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

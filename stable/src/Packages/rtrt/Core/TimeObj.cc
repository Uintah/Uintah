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



#include <Packages/rtrt/Core/TimeObj.h>

#include <Core/Geometry/Vector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Parallel.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using SCIRun::Thread;
using SCIRun::Parallel;

using namespace rtrt;
using namespace std;

SCIRun::Persistent* timeObj_maker() {
  return new TimeObj();
}

// initialize the static member type_id
SCIRun::PersistentTypeID TimeObj::type_id("TimeObj", "Object", timeObj_maker);

TimeObj::TimeObj(double rate)
    : Object(0), rate(rate)
{
    cur=0;
}

TimeObj::~TimeObj()
{
}

void TimeObj::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
    objs[cur]->intersect(ray, hit, st, ppc);
}

void TimeObj::light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			      DepthStats* st, PerProcessorContext* ppc)
{
  objs[cur]->light_intersect(ray, hit, atten, st, ppc);
}

void TimeObj::softshadow_intersect(Light* light, Ray& ray, HitInfo& hit,
				   double dist, Color& atten, DepthStats* st,
				   PerProcessorContext* ppc)
{
    objs[cur]->softshadow_intersect(light, ray, hit, dist, atten, st, ppc);
}

void TimeObj::multi_light_intersect(Light* light, const Point& orig,
				  const Array1<Vector>& dirs,
				  const Array1<Color>& attens,
				  double dist,
				  DepthStats* st, PerProcessorContext* ppc)
{
    objs[cur]->multi_light_intersect(light, orig, dirs, attens,
				     dist, st, ppc);
}

Vector TimeObj::normal(const Point&, const HitInfo&)
{
    cerr << "Error: TimeObj normal should not be called!\n";
    return Vector(0,0,0);
}


void TimeObj::add(Object* obj)
{
    objs.add(obj);
}

void TimeObj::animate(double t, bool& changed)
{
    int n=(int)(t*rate);
    cur=n%objs.size();
    //cerr << "TimeObj::animate: cur = "<<cur;
    changed=true;
    objs[cur]->animate(t, changed);
    //cerr << " : objs[cur]->name_ = "<<objs[cur]->name_<<endl;
}

void TimeObj::collect_prims(Array1<Object*>& prims)
{
  //  cerr << "TimeObj::collect_prims: objs.size() = "<<objs.size()<<endl;
    for(int i=0;i<objs.size();i++){
	objs[i]->collect_prims(prims);
    }
}

void TimeObj::parallel_preprocess(int proc)
{
  int start = objs.size()*proc/num_processors;
  int end = objs.size()*(proc+1)/num_processors;
  double max_radius=0;
  int pp_offset=0;
  int scratchsize=0;
  for(int i=start;i<end;i++){
    objs[i]->preprocess(max_radius, pp_offset, scratchsize);
  }
}

#if 0
void TimeObj::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
  //Parallel<TimeObj> phelper(this, &parallel_preprocess);
  num_processors=objs.size();
  if(num_processors>8)
    num_processors=8;
  Thread::parallel(this, &TimeObj::parallel_preprocess, num_processors, true);
  //Thread::parallel(phelper, num_processors, true);
}
#else
void TimeObj::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->preprocess(maxradius, pp_offset, scratchsize);
    }
}
#endif

void TimeObj::compute_bounds(BBox& bbox, double offset)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->compute_bounds(bbox, offset);
    }
}

const int TIMEOBJ_VERSION = 1;

void 
TimeObj::io(SCIRun::Piostream &str)
{
  str.begin_class("TimeObj", TIMEOBJ_VERSION);
  Object::io(str);
  SCIRun::Pio(str, cur);
  SCIRun::Pio(str, objs);
  SCIRun::Pio(str, rate);
  SCIRun::Pio(str, num_processors);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::TimeObj*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::TimeObj::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::TimeObj*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

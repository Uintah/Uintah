
#include <Packages/rtrt/Core/TimeObj.h>

#include <Core/Geometry/Vector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Parallel.h>

#include <iostream>

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

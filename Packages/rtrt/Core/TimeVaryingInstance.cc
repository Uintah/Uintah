
/*******************************************************************************\
 *                                                                             *
 * filename: TimeVaryingInstance.h                                             *
 * author  : R. Keith Morley                                                   *
 * last mod: 07/12/02                                                          *
 *                                                                             *
\*******************************************************************************/ 

#include <Packages/rtrt/Core/TimeVaryingInstance.h>
#include <Core/Geometry/Transform.h>

using namespace rtrt;
using namespace SCIRun;

SCIRun::Persistent* tvi_maker() {
  return new TimeVaryingInstance();
}
SCIRun::Persistent* f1_maker() {
  return new FishInstance1();
}
SCIRun::Persistent* f2_maker() {
  return new FishInstance2();
}

// initialize the static member type_id
SCIRun::PersistentTypeID TimeVaryingInstance::type_id("TimeVaryingInstance", 
						      "Instance", tvi_maker);
SCIRun::PersistentTypeID FishInstance1::type_id("FishInstance1", 
						"TimeVaryingInstance", 
						f1_maker);
SCIRun::PersistentTypeID FishInstance2::type_id("FishInstance2", 
						"TimeVaryingInstance", 
						f2_maker);


/*******************************************************************************/
TimeVaryingInstance::TimeVaryingInstance (InstanceWrapperObject* obj)
   : Instance(obj, new Transform())
{
  
   currentTransform->load_identity(); 
   obj->compute_bounds(bbox_orig,0);
   

   ocenter = bbox_orig.center();
   originToCenter = (Vector(ocenter.x(), ocenter.y(), 0));
   originToCenter.normalize();
   //bbox.extend(Point(-6, -6, 1));
   //bbox.extend(Point(6, 6, 2.5));
}

/*******************************************************************************/
void 
TimeVaryingInstance::computeTransform(double t)
{
   double vertScale = .25;
   double horizScale =.2;

   currentTransform->load_identity();
   // horizontal shift
   currentTransform->pre_translate(horizScale * originToCenter * sin(t));
   // rotate around z axis   
   currentTransform->pre_rotate(-(t * M_PI / 15), Vector(0, 0, 1));
   // vertical shift
   currentTransform->pre_translate(Vector(0, 0, vertScale * sin(.3*t)));

}

/*******************************************************************************/
void TimeVaryingInstance::compute_bounds(BBox& b, double /*offset*/)
{
   b.extend(bbox);
}

/*******************************************************************************/
void TimeVaryingInstance::animate(double t, bool& changed)
{
  changed = true;
  o->animate(t, changed);
  computeTransform(t); 
}

/*******************************************************************************/
void TimeVaryingInstance::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
                                       PerProcessorContext* ppc)
{
   double min_t = hit.min_t;

   if (!bbox.intersect(ray, min_t)) return;

   Ray tray;

   ray.transform(currentTransform,tray);
   Vector td = tray.direction();
   double scale = td.normalize();
   tray.set_direction(td);

   HitInfo thit;
   if (hit.was_hit) thit.min_t = hit.min_t * scale;

   o->intersect(tray,thit,st,ppc);

   // if the ray hit one of our objects....
   if (thit.was_hit)
   {
      min_t = thit.min_t / scale;
      if(hit.hit(this, min_t))
      {
         InstanceHit* i = (InstanceHit*)(hit.scratchpad);
         Point p = tray.origin() + thit.min_t*tray.direction();
         i->normal = thit.hit_obj->normal(p,thit);
         i->obj = thit.hit_obj;
         UVMapping * theUV = thit.hit_obj->get_uvmapping();
         theUV->uv(i->uv, p, thit );
      }
   }
}

/*******************************************************************************/
FishInstance1::FishInstance1(InstanceWrapperObject* obj, double _vertHeightScale, 
                             double _horizHeightScale, double _vertPerScale, 
                             double _horizPerScale, double _rotPerSec, double _startTime,
			     double _vertShift)
   : TimeVaryingInstance(obj), rotPerSec(_rotPerSec / 2.0),
     vertHeightScale(_vertHeightScale), horizHeightScale(_horizHeightScale), 
     horizPerScale(_horizPerScale), vertPerScale(_vertPerScale),
     startTime(_startTime), vertShift(_vertShift)
{
   currentTransform->load_identity();
   bbox.extend(Point(-6, -6,  vertShift + .5));
   bbox.extend(Point(6, 6,  vertShift + 2.5));
}

/*******************************************************************************/
void
FishInstance1::computeTransform(double t)
{
   double time = t + startTime;
   currentTransform->load_identity();
   // horizontal shift
   currentTransform->pre_translate(horizHeightScale * originToCenter * sin( horizPerScale * time));
   // rotate around z axis   
   currentTransform->pre_rotate(-(time * M_PI * rotPerSec), Vector(0, 0, 1));
   // vertical shift
   currentTransform->pre_translate(Vector(0, 0, vertShift + vertHeightScale * sin(vertPerScale * time)));

}

/*******************************************************************************/
FishInstance2::FishInstance2(InstanceWrapperObject* obj, double _vertHeightScale,
                             double _horizHeightScale, double _vertPerScale,
                             double _horizPerScale, double _rotPerSec, double _startTime,
                             double _vertShift)
   : TimeVaryingInstance(obj), rotPerSec(_rotPerSec / 2.0),
     vertHeightScale(_vertHeightScale), horizHeightScale(_horizHeightScale),
     horizPerScale(_horizPerScale), vertPerScale(_vertPerScale),
     startTime(_startTime), vertShift(_vertShift)
{
   currentTransform->load_identity();
   bbox.extend(Point(-6, 2,  vertShift + .5));
   bbox.extend(Point(6, 9,  vertShift + 2.5));
}

/*******************************************************************************/
void
FishInstance2::computeTransform(double t)
{
   double time = t + startTime;
   currentTransform->load_identity();
   // horizontal shift
   currentTransform->pre_translate(horizHeightScale * originToCenter * sin( horizPerScale * time));
   // rotate around z axis   
   currentTransform->pre_rotate((time * M_PI * rotPerSec), Vector(0, 0, 1));
   // scale in x
   currentTransform->pre_scale(Vector(1, .3, 1));
   // move towards north tube
   currentTransform->pre_translate(Vector(0, 7, 0));
   // vertical shift
   currentTransform->pre_translate(Vector(0, 0, vertShift + vertHeightScale * sin(vertPerScale * time)));

}

const int TimeVaryingInstance_VERSION = 1;
const int FishInstance1_VERSION = 1;
const int FishInstance2_VERSION = 1;

void 
TimeVaryingInstance::io(SCIRun::Piostream &str)
{
  str.begin_class("TimeVaryingInstance", TimeVaryingInstance_VERSION);
  Instance::io(str);
  SCIRun::Pio(str, center);
  SCIRun::Pio(str, ocenter);
  SCIRun::Pio(str, axis);
  SCIRun::Pio(str, rate);
  SCIRun::Pio(str, bbox_orig);
  SCIRun::Pio(str, originToCenter);
  str.end_class();
}

void 
FishInstance1::io(SCIRun::Piostream &str)
{
  str.begin_class("FishInstance1", FishInstance1_VERSION);
  TimeVaryingInstance::io(str);
  SCIRun::Pio(str, vertHeightScale);
  SCIRun::Pio(str, vertPerScale);
  SCIRun::Pio(str, horizHeightScale);
  SCIRun::Pio(str, horizPerScale);
  SCIRun::Pio(str, rotPerSec);
  SCIRun::Pio(str, startTime);
  SCIRun::Pio(str, vertShift);
  str.end_class();
}

void 
FishInstance2::io(SCIRun::Piostream &str)
{
  str.begin_class("FishInstance2", FishInstance2_VERSION);
  TimeVaryingInstance::io(str);
  SCIRun::Pio(str, vertHeightScale);
  SCIRun::Pio(str, vertPerScale);
  SCIRun::Pio(str, horizHeightScale);
  SCIRun::Pio(str, horizPerScale);
  SCIRun::Pio(str, rotPerSec);
  SCIRun::Pio(str, startTime);
  SCIRun::Pio(str, vertShift);
  str.end_class();
}

namespace SCIRun {

void Pio( Piostream& stream, rtrt::TimeVaryingInstance*& obj )
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::TimeVaryingInstance::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::TimeVaryingInstance*>(pobj);
    //ASSERT(obj != 0)
  }
}
void Pio( Piostream& stream, rtrt::FishInstance1*& obj )
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::FishInstance1::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::FishInstance1*>(pobj);
    //ASSERT(obj != 0)
  }
}
void Pio( Piostream& stream, rtrt::FishInstance2*& obj )
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::FishInstance2::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::FishInstance2*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun


















#include <Packages/rtrt/Core/Satellite.h>

extern double ORBIT_SPEED;
extern double ROTATE_SPEED;
extern bool   HOLO_ON;

namespace rtrt {
using namespace SCIRun;

Persistent* satellite_maker() {
  return new Satellite();
}

// initialize the static member type_id
PersistentTypeID Satellite::type_id("Satellite", "Object", satellite_maker);

void Satellite::animate(double t, bool& changed)
{
  changed = false;

  if (ORBIT_SPEED==0) {
    theta_ = 0;
    cen = Point(orb_radius_*cos(theta_),orb_radius_*sin(theta_),0);
    if (parent_)
      cen += (parent_->get_center().asVector());    
  }
  
  // orbit
  if (orb_speed_) {
    theta_ += orb_speed_*t*ORBIT_SPEED;
    if (theta_>628318.53) theta_=0; /* start over after 200,000 PI */
    cen = Point(orb_radius_*cos(theta_),orb_radius_*sin(theta_),0);
    if (parent_)
      cen += (parent_->get_center().asVector());
    changed = true;
  } 
  
  // revolution
  if (rev_speed_) {
    xform.load_identity();
    xform.pre_translate(-cen.asVector());
    xform.rotate(Vector(0,0,1),up);
    xform.pre_rotate(-rev_speed_*t*ROTATE_SPEED,Vector(0,0,1));
    xform.pre_scale(Vector(1./radius, 1./radius, 1./radius));
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, radius));
    ixform.pre_rotate(rev_speed_*t*ROTATE_SPEED,Vector(0,0,1));
    ixform.rotate(up, Vector(0,0,1));
    ixform.pre_translate(cen.asVector());
    changed = true;      
  }
}

const int SATELLITE_VERSION = 1;

void 
Satellite::io(SCIRun::Piostream &str)
{
  str.begin_class("Satellite", SATELLITE_VERSION);
  UVSphere::io(str);
  SCIRun::Pio(str, parent_);
  SCIRun::Pio(str, rev_speed_);
  SCIRun::Pio(str, orb_radius_);
  SCIRun::Pio(str, orb_speed_);
  SCIRun::Pio(str, theta_);
  str.end_class();
}
} // end namespace

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Satellite*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Satellite::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Satellite*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

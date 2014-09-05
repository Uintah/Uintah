
#include <Packages/rtrt/Core/Satellite.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>

namespace rtrt {
  // These are found in Gui.cc
  extern double ORBIT_SPEED;
  extern double ROTATE_SPEED;
  
  using namespace SCIRun;
  
  Persistent* satellite_maker() {
    return new Satellite();
  }
  
  // initialize the static member type_id
  PersistentTypeID Satellite::type_id("Satellite", "Object", satellite_maker);
}

using namespace rtrt;


Satellite::Satellite(const string &name, Material *mat, const Point &center,
                     double radius, double orb_radius, const Vector &up, 
                     Satellite *parent) 
  : UVSphere(mat, center, radius, up), parent_(parent), 
    rev_speed_(1), orb_radius_(orb_radius), orb_speed_(1)
{
  //theta_ = drand48()*6.282;
  theta_ = 0;
  Names::nameObject(name, this);
  
  if (orb_radius_ && parent_) {
    cen = parent->get_center();
    cen += Vector(orb_radius_*cos(theta_),
                  orb_radius_*sin(theta_),0);
  }
}

void Satellite::compute_bounds(BBox& bbox, double offset) {
#if _USING_GRID2_
  bbox.extend(cen,radius+offset);
#else
  if (parent_) {
    Point center = parent_->get_center();
    bbox.extend(center);
    Point extent = 
      Point(center.x()+parent_->get_orb_radius()+orb_radius_+radius+offset,
            center.y()+parent_->get_orb_radius()+orb_radius_+radius+offset,
            center.z()+radius+offset);
    bbox.extend( extent );
    extent = 
      Point(center.x()-(parent_->get_orb_radius()+orb_radius_+radius+offset),
            center.y()-(parent_->get_orb_radius()+orb_radius_+radius+offset),
            center.z()-(radius+offset));
    bbox.extend( extent );
    
  } else {
    bbox.extend(cen, orb_radius_+radius+offset);
  }
#endif
}
  
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

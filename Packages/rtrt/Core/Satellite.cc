
#include <Packages/rtrt/Core/Satellite.h>

extern double ORBIT_SPEED;
extern double ROTATE_SPEED;
extern bool   HOLO_ON;

namespace rtrt {

void Satellite::animate(double t, bool& changed)
{
  changed = false;
  
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
    xform.rotate(right, Vector(1,0,0));
    xform.rotate(up, Vector(0,0,1));
    xform.pre_rotate(-rev_speed_*t*ROTATE_SPEED,Vector(0,0,1));
    xform.pre_scale(Vector(1./radius, 1./radius, 1./radius));
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, radius));
    ixform.pre_rotate(rev_speed_*t*ROTATE_SPEED,Vector(0,0,1));
    ixform.rotate(Vector(0,0,1), up);
    ixform.rotate(Vector(1,0,0), right);
    ixform.pre_translate(cen.asVector());
    changed = true;      
  }
}

} // end namespace

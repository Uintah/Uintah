
#ifndef SATELLITE_H
#define SATELLITE_H 1

#include <Packages/rtrt/Core/UVSphere.h>
#include <stdlib.h>

namespace rtrt {

class Satellite : public UVSphere
{

 protected:

  Satellite *parent_;
  double    rev_speed_;
  double    orb_radius_;
  double    orb_speed_;
  double    theta_;

 public:

  Satellite(const string &name, Material *mat, const Point &center,
            double radius, double orb_radius, const Vector &up=Vector(0,0,1), 
            Satellite *parent=0) 
    : UVSphere(mat, center, radius, up), parent_(parent), 
    rev_speed_(1), orb_radius_(orb_radius), orb_speed_(1)
  {
    theta_ = drand48()*6.282;
    name_ = name;

    if (orb_radius_ && parent_) {
      cen = parent->get_center();
      cen += Vector(orb_radius_*cos(theta_),
                    orb_radius_*sin(theta_),0);
    }
  }
  virtual ~Satellite() {}

  Satellite *get_parent() { return parent_; }
  void set_parent(Satellite *p) { parent_ = p; }

  double get_rev_speed() const { return rev_speed_; }
  void set_rev_speed(double speed) { rev_speed_ = speed; }

  double get_orb_speed() const { return orb_speed_; } 
  void set_orb_speed(double speed) { orb_speed_ = speed; }

  double get_orb_radius() const { return orb_radius_; } 
  void set_orb_radius(double radius) { orb_radius_ = radius; }

  double get_radius() const { return radius; }
  void set_radius(double r) { radius = r; }

  Point &get_center() { return cen; }
  void set_center(const Point &p) { cen = p; }

  string get_name() const { return name_; }
  void set_name(const string &s) { name_ = s; }

  virtual void compute_bounds(BBox& bbox, double offset)
  {
    if (parent_) {
      parent_->compute_bounds(bbox,offset);
      bbox.extend(parent_->get_center(), 
                  parent_->get_orb_radius()+parent_->get_radius()+
                  orb_radius_+radius+offset);
    } else {
      bbox.extend(cen, orb_radius_+radius+offset);
    }
  }

  virtual void animate(double t, bool& changed)
  {
    changed = false;

    // orbit
    if (orb_speed_) {
      theta_ += orb_speed_*t;
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
      xform.pre_rotate(-rev_speed_*t,Vector(0,0,1));
      xform.pre_scale(Vector(1./radius, 1./radius, 1./radius));
      ixform.load_identity();
      ixform.pre_scale(Vector(radius, radius, radius));
      ixform.pre_rotate(rev_speed_*t,Vector(0,0,1));
      ixform.rotate(Vector(0,0,1), up);
      ixform.rotate(Vector(1,0,0), right);
      ixform.pre_translate(cen.asVector());
      changed = true;      
    }
  }
};

} // end namespace

#endif

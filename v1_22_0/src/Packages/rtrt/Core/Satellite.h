
#ifndef SATELLITE_H
#define SATELLITE_H 1

#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/Names.h>
#include <stdlib.h>

namespace rtrt {
class Satellite;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Satellite*&);
}

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
    //theta_ = drand48()*6.282;
    theta_ = 0;
    Names::nameObject(name, this);

    if (orb_radius_ && parent_) {
      cen = parent->get_center();
      cen += Vector(orb_radius_*cos(theta_),
                    orb_radius_*sin(theta_),0);
    }
  }
  virtual ~Satellite() {}
  Satellite() : UVSphere() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Satellite*&);

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

  virtual void compute_bounds(BBox& bbox, double offset)
  {
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

  virtual void animate(double t, bool& changed);
};

} // end namespace

#endif

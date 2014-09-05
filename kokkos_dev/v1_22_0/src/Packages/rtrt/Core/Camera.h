

#ifndef CAMERA_H
#define CAMERA_H 1

#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class Camera;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Camera*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

class Ray;
class Stealth;
class Scene;
class PerProcessorContext;
class Dpy;
class Gui;

class Camera : public SCIRun::Persistent {

protected:

  friend class Dpy;
  friend class Gui;

  char pad1[128];
  Point eye;
  Point lookat;
  Vector up;

  double fov;
  double verticalFov_;
  double windowAspectRatio_;

  Vector u, v;
  Vector uhat, vhat, what;
  Vector direction;
  double eyesep;

  // This is the ammount that the primary rays will be offset.
  double ray_offset;
  
  char pad2[128];
  
public:
  Camera(const Point& eye, const Point& lookat,
	 const Vector& up, double fov, double ray_offset = 0);
  Camera();
  virtual ~Camera();


  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Camera*&);

  virtual void makeRay(Ray& ray, double x, double y, double xres, double yres);
  virtual void makeRayL(Ray& ray, double x, double y, double xres, 
			double yres);
  virtual void makeRayR(Ray& ray, double x, double y, double xres, 
			double yres);
  void get_viewplane(Vector& u, Vector& v) const;
  virtual void setup();
  void print();

  inline const Point& get_eye() const { return eye; }
  // Returns 2 points on either side of the camera at 2*separation distance.
  void get_ears( Point & left, Point & right, double separation ) const;

  inline bool operator != (const Camera& c) const {
    return eye != c.eye || lookat != c.lookat || up != c.up || fov != c.fov || verticalFov_ != c.verticalFov_ || ray_offset != c.ray_offset || eyesep != c.eyesep;
  }

  void set_lookat(const Point&);
  Point get_lookat() const;

  void set_fov(double fov);
  double get_fov() const;
  void setWindowAspectRatio( double ratio );

  double get_eyesep() const;
  void set_eye(const Point&);
  void scale_eyesep(double scale);
  void set_up(const Vector&);
  Vector get_up() const;
  void getParams(Point& origin, Vector& lookdir,
		 Vector& up, Vector& side, double& fov);

  void updatePosition( Stealth & stealth, 
		       Scene * scene, PerProcessorContext * ppc );

  void followPath( Stealth & stealth );
  void flatten(); // reset pitch to 0 and roll to 0.(note: no roll currently)

  void set_ray_offset(double off) { ray_offset = off; }
  double get_ray_offset() { return ray_offset; }

  // Location/Look At update functions that most likely will
  // (should) be called by the spaceball input:
  void moveForwardOrBack( double amount );
  void moveVertically( double amount );
  void moveLaterally( double amount );
  void changePitch( double amount );
  void changeFacing( double amount );

};

} // end namespace rtrt

#endif

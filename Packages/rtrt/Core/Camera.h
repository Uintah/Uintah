

#ifndef CAMERA_H
#define CAMERA_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

class Ray;
class Stealth;
class Scene;
class PerProcessorContext;
class Dpy;
class Gui;

class Camera {

protected:

  friend class Dpy;
  friend class Gui;

  char pad1[128];
  Point eye;
  Point lookat;
  Vector up;
  double fov;
  Vector u, v;
  Vector uhat, vhat, what;
  Vector direction;
  double eyesep;
  
  char pad2[128];
  
public:
    Camera(const Point& eye, const Point& lookat,
	   const Vector& up, double fov);
    Camera();
    virtual ~Camera();
    virtual void makeRay(Ray& ray, double x, double y, double xres, double yres);
    virtual void makeRayL(Ray& ray, double x, double y, double xres, double yres);
    virtual void makeRayR(Ray& ray, double x, double y, double xres, double yres);
    void get_viewplane(Vector& u, Vector& v) const;
    virtual void setup();
    void print();
    inline const Point& get_eye() const {
	return eye;
    }
    inline bool operator != (const Camera& c) const {
	return eye != c.eye || lookat != c.lookat || up != c.up || fov != c.fov || eyesep != c.eyesep;
    }
    void set_lookat(const Point&);
    Point get_lookat() const;
    void set_fov(double fov);
    double get_fov() const;
    double get_eyesep() const;
    void set_eye(const Point&);
    void scale_eyesep(double scale);
    void set_up(const Vector&);
    Vector get_up() const;
    void getParams(Point& origin, Vector& lookdir,
		   Vector& up, double& fov);

    void updatePosition( Stealth & stealth, 
			 Scene * scene, PerProcessorContext * ppc );

    void followPath( Stealth & stealth );
    void flatten(); // reset pitch to 0 and roll to 0.(note: no roll currently)

};

} // end namespace rtrt

#endif



#ifndef CAMERA_H
#define CAMERA_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {

  using namespace SCIRun;
  
class Ray;

class Camera {
    friend class Dpy;
    char pad1[128];
    Point eye;
    Point lookat;
    Vector up;
    double fov;
    Vector u,v;
    Vector direction;
    Vector eyesep;
    char pad2[128];
public:
    Camera(const Point& eye, const Point& lookat,
	   const Vector& up, double fov);
    Camera();
    ~Camera();
    void makeRay(Ray& ray, double x, double y, double xres, double yres);
    void makeRayL(Ray& ray, double x, double y, double xres, double yres);
    void makeRayR(Ray& ray, double x, double y, double xres, double yres);
    void get_viewplane(Vector& u, Vector& v);
    void setup();
    void print();
    inline const Point& get_eye() const {
	return eye;
    }
    inline bool operator != (const Camera& c) const {
	return eye != c.eye || lookat != c.lookat || up != c.up || fov != c.fov;
    }
    void set_lookat(const Point&);
    Point get_lookat() const;
    void set_fov(double fov);
    double get_fov() const;
    void set_eye(const Point&);
    void set_up(const Vector&);
    Vector get_up() const;
    void getParams(Point& origin, Vector& lookdir,
		   Vector& up, double& fov);
};

} // end namespace rtrt

#endif

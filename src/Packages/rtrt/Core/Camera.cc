
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Core/Geometry/Transform.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

Camera::Camera()
{
}

Camera::Camera(const Point& eye, const Point& lookat,
	       const Vector& up, double fov)
  : eye(eye), lookat(lookat), up(up), fov(fov), eyesep(1)
{
    setup();
}

Camera::~Camera()
{
}

#if 1
void Camera::makeRay(Ray& ray, double x, double y, double, double iyres)
{
    ray.set_origin(eye);
    double screenx=(x+0.5)*iyres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
}
#endif
void Camera::makeRayL(Ray& ray, double x, double y, double, double iyres)
{
    ray.set_origin(eye-v*5*eyesep*iyres);
    double screenx=(x+0.5)*iyres-0.5;
    Vector sv(v*screenx+v*5*iyres*eyesep);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
}
void Camera::makeRayR(Ray& ray, double x, double y, double, double iyres)
{
    ray.set_origin(eye+v*5*eyesep*iyres);
    double screenx=(x+0.5)*iyres-0.5;
    Vector sv(v*screenx-v*5*iyres*eyesep);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
}

void Camera::get_viewplane(Vector& uu, Vector& vv)
{
    uu=u;
    vv=v;
}

void Camera::setup()
{
    direction=lookat-eye;
    double dist=direction.length();
    v=Cross(direction, up);
    if(v.length2() == 0.0){
	cerr << "Ambiguous up direciton...\n";
    }
    v.normalize();

    u=Cross(v, direction);
    u.normalize();

    double height=2.0*dist*tan(fov*0.5*M_PI/180.);
    u*=height;
    double width=2.0*dist*tan(fov*0.5*M_PI/180.);
    v*=width;

}

void Camera::print()
{
  cerr << "-eye " << eye.x() << ' ' << eye.y() << ' ' << eye.z() << ' ';
  cerr << "-lookat " << lookat.x() << ' ' << lookat.y() << ' ' << lookat.z() << ' ';
  cerr << "-up " << up.x() << ' ' << up.y() << ' ' << up.z() << ' ';
  cerr << "-fov " << fov << '\n';
}

void Camera::set_eye(const Point& e)
{
    eye=e;
}

void Camera::scale_eyesep(double scale)
{
  eyesep*=scale;
}

void Camera::set_up(const Vector& u)
{
    up=u;
}

Vector Camera::get_up() const
{
    return up;
}

void Camera::set_lookat(const Point& l)
{
    lookat=l;
}

Point Camera::get_lookat() const
{
    return lookat;
}

void Camera::set_fov(double f)
{
    fov=f;
}

double Camera::get_eyesep() const
{
  return eyesep;
}

double Camera::get_fov() const
{
    return fov;
}

void Camera::getParams(Point& origin, Vector& direction,
		       Vector& up, double& vfov)
{
    origin=eye;
    direction=this->direction;
    up=u;
    vfov=fov;
}

void
Camera::followPath( Stealth & stealth )
{
  stealth.getNextLocation( eye, lookat);
  setup();
}

void
Camera::updatePosition( Stealth & stealth )
{
  Vector forward( direction );
  Vector theUp( u );
  Vector side( v );

  forward.normalize();
  theUp.normalize();
  side.normalize();

  double speed;

  // Move the eyepoint based on direction(s) of movement (up, right, forward)
  eye += forward * stealth.getSpeed(0);
  eye += side    * stealth.getSpeed(1);
  eye += theUp   * stealth.getSpeed(2);

  // Move the lookat point correspondingly.
  lookat += forward * stealth.getSpeed(0);
  lookat += side    * stealth.getSpeed(1);
  lookat += theUp   * stealth.getSpeed(2);

  // Pitching
  if( ( speed = stealth.getSpeed(3) ) != 0 )
    {
      // Keeps you from pitching up or down completely!
      Transform t;
      t.post_translate( Vector(eye) );
      t.post_rotate( speed/20, side );
      t.post_translate( Vector(-eye) );

      Point old_lookat = lookat;
      Point new_lookat = t.project( lookat );

      lookat = new_lookat;
      setup();

      Vector new_forward( direction );
      new_forward.normalize();

      if( Dot( new_forward, up ) > .9 || Dot( new_forward, up ) < -.9 )
	{
	  // We have pitched up or down too much.  Reset eye to before
	  // this last pitch adjustment.  Tell stealth to stop pitching.
	  lookat = old_lookat;
	  setup();
	  stealth.stopPitching();
	}
    }

  // Rotating
  if( ( speed = stealth.getSpeed(4) ) != 0 )
    {
      Transform t;
      t.post_translate( Vector(eye) );
      t.post_rotate( -speed/20, theUp );
      t.post_translate( Vector(-eye) );
      lookat = t.project( lookat );
      setup();
    }
}


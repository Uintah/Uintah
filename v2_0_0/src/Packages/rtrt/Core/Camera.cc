
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Geometry/Transform.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

Persistent* camera_maker() {
  return new Camera();
}

// initialize the static member type_id
PersistentTypeID Camera::type_id("Camera", "Persistent", camera_maker);

Camera::Camera()
{
}

Camera::Camera(const Point& eye, const Point& lookat,
	       const Vector& up, double fov, double ray_offset) :
  eye(eye), lookat(lookat), up(up), fov(fov), 
  verticalFov_(fov), ray_offset(ray_offset), windowAspectRatio_(1.0),
  eyesep(1)
{
  setup();
}

Camera::~Camera()
{
}

void Camera::makeRay(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
    ray.set_origin(eye + raydir*ray_offset);
}

void Camera::makeRayL(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx+v*5*iyres*eyesep);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir + v*5*eyesep*iyres + raydir*ray_offset);
    ray.set_origin(eye - v*5*eyesep*iyres + raydir*ray_offset);
}

void Camera::makeRayR(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx-v*5*iyres*eyesep);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir - v*5*eyesep*iyres + raydir*ray_offset);
    ray.set_origin(eye + v*5*eyesep*iyres + raydir*ray_offset);
}

void Camera::get_viewplane(Vector& uu, Vector& vv) const
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

    double height=2.0*dist*tan(verticalFov_*0.5*M_PI/180.0);
    u*=height;
    double width=2.0*dist*tan(fov*0.5*M_PI/180.0);
    v*=width;

}

void
Camera::get_ears( Point & left, Point & right, double separation ) const
{
  Vector side = v;
  side.normalize();

  left  = eye +  (side * separation);
  right = eye + -(side * separation);
}

void
Camera::print()
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

void
Camera::setWindowAspectRatio( double ratio )
{
  windowAspectRatio_ = ratio;
  verticalFov_ = fov * ratio;
  setup();
}

void
Camera::set_fov(double f)
{
    fov=f;
    verticalFov_ = fov * windowAspectRatio_;
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
		       Vector& up, Vector& side, double& theFov)
{
    origin=eye;
    direction=this->direction;
    up=u;
    side=v;
    theFov=fov;
}

void
Camera::followPath( Stealth & stealth )
{
  
  stealth.getNextLocation( eye, lookat);
  setup();
}

void
Camera::flatten() // reset pitch to 0 and roll to 0.(note: no roll currently)
{
  lookat.z( eye.z() );   // Clears Pitch
  up = Vector( 0, 0, .999 ); // Clears Roll
  setup();
}

void
Camera::updatePosition( Stealth & stealth, 
			Scene * scene, PerProcessorContext * ppc )
{
  Vector forward( direction );
  Vector theUp( u );
  Vector side( v );

  forward.normalize();
  theUp.normalize();
  side.normalize();

  double rotational_speed_damper = 100;

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
      t.post_rotate( speed/rotational_speed_damper, side );
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
	  stealth.stopPitch();
	}
    }

  // Rotating
  if( ( speed = stealth.getSpeed(4) ) != 0 )
    {
      Transform t;
      t.post_translate( Vector(eye) );
      t.post_rotate( -speed/rotational_speed_damper, theUp );
      t.post_translate( Vector(-eye) );
      lookat = t.project( lookat );
      setup();
    }

  // After updating position based on stealth, update based on gravity
  if( stealth.gravityIsOn() ) {

    HitInfo    hit;
    DepthStats ds;
    Ray        r( eye, -up ); // ray pointing straight downward
    Object   * obj = scene->get_object();

    obj->intersect( r, hit, &ds, ppc );

    double time = hit.min_t;

    // I want to stay this far from the ground:
    // specific to range-400
    double height_off_ground = 5;


    if( time < MAXDOUBLE ) // Ie: ray hit the ground
      {
	double gravity = stealth.getGravityForce();

	if( ( time > height_off_ground ) &&
	    fabs(time - height_off_ground) > gravity )
	  {
	    cout << "going down: time: " << time << "\n";
	    // Move downward
	    eye    -= up * gravity;
	    lookat -= up * gravity;
	  }
	else if( time < height_off_ground ) 
	  { // move upwards to maintain constant distance from ground
	    eye    += up * gravity;
	    lookat += up * gravity;
	    cout << "going up: time: " << time << "\n";
	  }
      }
  }

  // Move the lights that are fixed to the eye
  for(int i = 0; i < scene->nlights(); i++) {
    Light *light = scene->light(i);
    if (light->fixed_to_eye) {
//      light->updatePosition(eye, side, theUp, forward);
      light->updatePosition(eye, Vector(side*light->eye_offset_basis.x()+
					theUp*light->eye_offset_basis.y()+
					forward*light->eye_offset_basis.z()),
			    forward);
    }
  }
}

void
Camera::moveForwardOrBack( double amount )
{
  Vector forward( direction );
  forward.normalize();

  eye    += forward * amount;
  lookat += forward * amount;
  setup();
}

void
Camera::moveVertically( double amount )
{
  Vector theUp( u );
  theUp.normalize();

  eye    += theUp * amount;
  lookat += theUp * amount;
  setup();
}

void
Camera::moveLaterally( double amount )
{
  Vector side( v );
  side.normalize();

  eye    += side * amount;
  lookat += side * amount;
  setup();
}

void
Camera::changePitch( double amount )
{
  Vector side( v );
  side.normalize();

  Transform t;
  t.post_translate( Vector(eye) );
  t.post_rotate( amount, side );
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
    }
}

void
Camera::changeFacing( double amount )
{
  Vector theUp( u );
  Transform t;

  theUp.normalize();

  t.post_translate( Vector(eye) );
  t.post_rotate( amount, theUp );
  t.post_translate( Vector(-eye) );

  lookat = t.project( lookat );
  setup();
}

const int CAMERA_VERSION = 1;

void 
Camera::io(SCIRun::Piostream &str)
{
  str.begin_class("Camera", CAMERA_VERSION);
  SCIRun::Pio(str, eye);
  SCIRun::Pio(str, lookat);
  SCIRun::Pio(str, up);
  SCIRun::Pio(str, fov);
  SCIRun::Pio(str, u);
  SCIRun::Pio(str, v);
  SCIRun::Pio(str, uhat);
  SCIRun::Pio(str, vhat);
  SCIRun::Pio(str, what);
  SCIRun::Pio(str, direction);
  SCIRun::Pio(str, eyesep);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Camera*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Camera::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Camera*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

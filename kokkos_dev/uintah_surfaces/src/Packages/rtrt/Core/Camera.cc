
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>

#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

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
  verticalFov_(fov), windowAspectRatio_(1.0),
  eyesep(1), ray_offset(ray_offset)
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

void Camera::makeRayLR(Ray& rayL, Ray& rayR, double x, double y,
                       double ixres, double iyres)
{
  double screenx=(x+0.5)*ixres-0.5;
  Vector sv(v*screenx);
  double screeny=(y+0.5)*iyres-0.5;
  Vector su(u*screeny);
  // Here direction is lookat-eye.  So doing a little bit of math gets us:
  Point neweyeL = eye-(v*5*ixres*eyesep);
  Point neweyeR = eye+(v*5*ixres*eyesep);
  Vector raydirL=(su+sv)+(lookat-neweyeL);
  Vector raydirR=(su+sv)+(lookat-neweyeR);
  raydirL.normalize();
  raydirR.normalize();
  rayL.set_direction(raydirL);
  rayR.set_direction(raydirR);
  rayL.set_origin(neweyeL + raydirL*ray_offset);
  rayR.set_origin(neweyeR + raydirR*ray_offset);
}  


#if 1
#if 0
void Camera::makeRayL(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx+v*5*ixres*eyesep);
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
    Vector sv(v*screenx-v*5*ixres*eyesep);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir - v*5*eyesep*iyres + raydir*ray_offset);
    ray.set_origin(eye + v*5*eyesep*iyres + raydir*ray_offset);
}
#else
// Here the idea is to actually change the location of the eye point in
// the plane parallel to the image plane.  The pixel location will
// remain the same, but direction will have to be recomputed.
void Camera::makeRayL(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    // Here direction is lookat-eye.  So doing a little bit of math gets us:
    Point neweye = eye-v*5*ixres*eyesep;
    Vector raydir=su+sv+(lookat-neweye);
    raydir.normalize();
    ray.set_direction(raydir);
    ray.set_origin(neweye + raydir*ray_offset);
}
void Camera::makeRayR(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    // Here direction is lookat-eye.  So doing a little bit of math gets us:
    Point neweye = eye+v*5*ixres*eyesep;
    Vector raydir=su+sv+(lookat-neweye);
    raydir.normalize();
    ray.set_direction(raydir);
    ray.set_origin(neweye + raydir*ray_offset);
}


#endif
#else
// Old school stereo, just move the eyepoint
void Camera::makeRayL(Ray& ray, double x, double y, double ixres, double iyres)
{
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
    ray.set_origin(eye-v*5*iyres);
}
void Camera::makeRayR(Ray& ray, double x, double y, double ixres, double iyres)
{
    ray.set_origin(eye+v*5*iyres);
    double screenx=(x+0.5)*ixres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
}
#endif

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

bool Camera::read(const char* buf) {
  bool call_setup = false;

  istringstream in(buf);
  string token;
  in >> token;

  while(in) {
    double v1, v2, v3;
    if (token == "-eye") {
      in >> v1 >> v2 >> v3;
      if (!in.fail()) {
        call_setup = true;
        eye = Point(v1, v2, v3);
        cerr << "Setting eye to "<<eye<<"\n";
      } else in.clear();
    } else if (token == "-lookat") {
      in >> v1 >> v2 >> v3;
      if (!in.fail()) {
        call_setup = true;
        lookat = Point(v1, v2, v3);
        cerr << "Setting lookat to "<<lookat<<"\n";
      } else in.clear();
    } else if (token == "-up") {
      in >> v1 >> v2 >> v3;
      if (!in.fail()) {
        call_setup = true;
        up = Vector(v1, v2, v3);
        cerr << "Setting up to "<<up<<"\n";
      } else in.clear();
    } else if (token == "-fov") {
      in >> v1;
      if (!in.fail()) {
        call_setup = true;
        set_fov(v1);
        cerr << "Setting fov to "<<fov<<"\n";
      } else in.clear();
    }
    in >> token;
  }
  if (call_setup) setup();
  return call_setup;
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
  
  stealth.getNextLocation( this );
  // This call to setup is redundant as getNextLocation calls it.
  //  setup();
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

Camera Camera::operator+(const Camera& right) const {
  Camera result;

  result.eye = (eye + right.eye).asPoint();
  result.lookat = (lookat + right.lookat).asPoint();
  result.up = up + right.up;

  result.fov = fov + right.fov;
  result.verticalFov_ = verticalFov_ + right.verticalFov_;
  result.windowAspectRatio_ = windowAspectRatio_ + right.windowAspectRatio_;

  result.eyesep = eyesep + right.eyesep;
  result.ray_offset = ray_offset + right.ray_offset;

  return result;
}

Camera Camera::operator-(const Camera& right) const {
  Camera result;

  result.eye = (eye - right.eye).asPoint();
  result.lookat = (lookat - right.lookat).asPoint();
  result.up = up - right.up;

  result.fov = fov - right.fov;
  result.verticalFov_ = verticalFov_ - right.verticalFov_;
  result.windowAspectRatio_ = windowAspectRatio_ - right.windowAspectRatio_;

  result.eyesep = eyesep - right.eyesep;
  result.ray_offset = ray_offset - right.ray_offset;

  return result;
}

Camera Camera::operator*(double val) const {
  Camera result;

  result.eye = eye * val;
  result.lookat = lookat * val;
  result.up = up * val;

  result.fov = fov * val;
  result.verticalFov_ = verticalFov_ * val;
  result.windowAspectRatio_ = windowAspectRatio_ * val;

  result.eyesep = eyesep * val;
  result.ray_offset = ray_offset * val;

  return result;
}

void Camera::transform(Transform t, TransformCenter center) {
  Vector cen;
  switch(center){
  case Eye:
    cen = eye.asVector();
    break;
  case LookAt:
    cen = lookat.asVector();
    break;
  case Origin:
    cen = Vector(0,0,0);
    break;
  }

  Vector lookdir(eye-lookat);
  double length = lookdir.length();
  Transform frame;
  frame.load_basis(Point(0,0,0), v.normal()*length, u.normal()*length, lookdir);
  frame.pre_translate(cen);
  //  double tmp = lookdir.length();

  Transform frame_inv(frame);
  frame_inv.invert();

  Transform t2;
  t2.load_identity();
  t2.pre_trans(frame_inv);
  t2.pre_trans(t);
  t2.pre_trans(frame);

  up = t2.project(up);
  eye = t2.project(eye);
  lookat = t2.project(lookat);
  setup();
}

void Camera::scaleFOV(double scale) {
  double fov_min = 0;
  double fov_max = 180;
  double tfov = RtoD(2*atan(scale*tan(DtoR(fov/2.))));
  tfov = Clamp(tfov, fov_min, fov_max);
  set_fov(tfov);
  setup();
}

void Camera::translate(Vector t)
{
  Vector trans(u*t.y()+v*t.x());

  eye += trans;
  lookat += trans;
  setup();
}

void Camera::dolly(double scale)
{
  Vector dir = lookat - eye;
  eye += dir*scale;
  setup();
}

void Camera::autoview(BBox& bbox, double new_fov) {
  if (new_fov > 0)
    set_fov(new_fov);

  Vector diag(bbox.diagonal());
  double w=diag.length();
  Vector lookdir(eye-lookat);
  lookdir.normalize();
  double scale = 1.0/(2*tan(DtoR(fov/2.0)));
  double length = w*scale;
  lookat = bbox.center();
  eye = lookat+lookdir*length;
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
  SCIRun::Pio(str, verticalFov_);
  SCIRun::Pio(str, windowAspectRatio_);
//   SCIRun::Pio(str, u);
//   SCIRun::Pio(str, v);
//   SCIRun::Pio(str, direction);
  SCIRun::Pio(str, eyesep);
  SCIRun::Pio(str, ray_offset);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Camera*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Camera::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Camera*>(pobj);
    obj->setup();
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

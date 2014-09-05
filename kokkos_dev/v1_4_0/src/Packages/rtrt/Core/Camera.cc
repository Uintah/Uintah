
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Ray.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

Camera::Camera()
{
}

Camera::Camera(const Point& eye, const Point& lookat,
	       const Vector& up, double fov)
    : eye(eye), lookat(lookat), up(up), fov(fov)
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
    ray.set_origin(eye-v*5*iyres);
    double screenx=(x+0.5)*iyres-0.5;
    Vector sv(v*screenx);
    double screeny=(y+0.5)*iyres-0.5;
    Vector su(u*screeny);
    Vector raydir=su+sv+direction;
    raydir.normalize();
    ray.set_direction(raydir);
}
void Camera::makeRayR(Ray& ray, double x, double y, double, double iyres)
{
    ray.set_origin(eye+v*5*iyres);
    double screenx=(x+0.5)*iyres-0.5;
    Vector sv(v*screenx);
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
    cerr << "eye=" << eye << '\n';
    cerr << "lookat=" << lookat << '\n';
    cerr << "up=" << up << '\n';
    cerr << "fov=" << fov << '\n';
}

void Camera::set_eye(const Point& e)
{
    eye=e;
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



/*
 *  View.cc:  The camera
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/View.h>
#include <Math/Trig.h>

View::View()
{
}

View::~View()
{
}

View::View(const Point& eyep, const Point& lookat, const Vector& up,
	   double fov)
: eyep_(eyep), lookat_(lookat), up_(up), fov_(fov)
{
}

View::View(const View& copy)
: eyep_(copy.eyep_), lookat_(copy.lookat_), up_(copy.up_), fov_(copy.fov_)
{
}

View& View::operator=(const View& copy)
{
    eyep_=copy.eyep_;
    lookat_=copy.lookat_;
    up_=copy.up_;
    fov_=copy.fov_;
    return *this;
}

void View::get_viewplane(double aspect, double zdist,
			 Vector& u, Vector& v)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=zdist*Tan(DtoR(fov()/2.))*2.;
    double yviewsize=xviewsize/aspect;
    x*=xviewsize;
    y*=yviewsize;
    u=x;
    v=y;
}

Point View::eyespace_to_objspace(const Point& ep, double aspect)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=Tan(DtoR(fov()/2.))*2.;
    double yviewsize=xviewsize/aspect;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;
    x*=xscale;
    y*=yscale;
    
    Point p(eyep()+x*ep.x()+y*ep.y()+z*ep.z());
    return p;
}

Point View::objspace_to_eyespace(const Point& ep, double aspect)
{
    // first compute basic cordiante frame

    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=Tan(DtoR(fov()/2.))*2.;
    double yviewsize=xviewsize/aspect;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;

    // the transform everything into eyespace
    
//    x*=xscale;
//    y*=yscale;
//    x.x(x.x()/xscale);
//    y.y(y.y()/yscale);
    Point p(Dot(x,ep-eyep_.vector())/xscale,
	    Dot(y,ep-eyep_.vector())/yscale,
	    Dot(z,ep-eyep_.vector()));
    return p;
}
Point View::eyespace_to_objspace_ns(const Point& ep, double aspect)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));
    
    Point p(eyep()+x*ep.x()+y*ep.y()+z*ep.z());
    return p;
}

Point View::objspace_to_eyespace_ns(const Point& ep, double aspect)
{
    // first compute basic cordiante frame

    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));

    Point p(Dot(x,ep-eyep_.vector()),
	    Dot(y,ep-eyep_.vector()),
	    Dot(z,ep-eyep_.vector()));
    return p;
}

double View::depth(const Point& p)
{
    Vector dir(lookat()-eyep());
    dir.normalize();
    double d=-Dot(eyep(), dir);
    double dist=Dot(p, dir)+d;
    return dist;
}

Point View::lookat() const
{
    return lookat_;
}

Point View::eyep() const
{
    return eyep_;
}

Vector View::up() const
{
    return up_;
}

double View::fov() const
{
    return fov_;
}

void View::eyep(const Point& e)
{
    eyep_=e;
}

void View::lookat(const Point& l)
{
    lookat_=l;
}

void View::fov(double f)
{
    fov_=f;
}

void View::up(const Vector& u)
{
    up_=u;
}


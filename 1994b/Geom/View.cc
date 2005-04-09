
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
: eyep(eyep), lookat(lookat), up(up), fov(fov)
{
}

View::View(const View& copy)
: eyep(copy.eyep), lookat(copy.lookat), up(copy.up), fov(copy.fov)
{
}

View& View::operator=(const View& copy)
{
    eyep=copy.eyep;
    lookat=copy.lookat;
    up=copy.up;
    fov=copy.fov;
    return *this;
}

void View::get_viewplane(double aspect, double zdist,
			 Vector& u, Vector& v)
{
    Vector lookdir(lookat-eyep);
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=zdist*Tan(DtoR(fov/2.))*2.;
    double yviewsize=xviewsize/aspect;
    x*=xviewsize;
    y*=yviewsize;
    u=x;
    v=y;
}

Point View::eyespace_to_objspace(const Point& ep, double aspect)
{
    Vector lookdir(lookat-eyep);
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=Tan(DtoR(fov/2.))*2.;
    double yviewsize=xviewsize/aspect;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;
    x*=xscale;
    y*=yscale;
    
    Point p(eyep+x*ep.x()+y*ep.y()+z*ep.z());
    return p;
}

double View::depth(const Point& p)
{
    Vector dir(lookat-eyep);
    dir.normalize();
    double d=-Dot(eyep, dir);
    double dist=Dot(p, dir)+d;
    return dist;
}

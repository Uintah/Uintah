/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <stdio.h>
#include <math.h>
#include <UI/View.h>
#include <Logging/Log.h>

namespace SemotusVisum {

View::View()
{
  // Set default initial values of all member variables to 0

  eyep_ = Point3d(0, 0, 0);
  lookat_ = Point3d(0, 0, 0);
  up_ = Vector(0, 0, 0);
  fov_ = 0;
}

View::~View()
{
}

View::View(const Point3d& eyep, const Point3d& lookat, const Vector& up,
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

int
View::operator==(const View& copy)
{
  if ( eyep_ == copy.eyep_ && lookat_ == copy.lookat_ &&
       up_ == copy.up_     && fov_ == copy.fov_           )
    return 1;
  else
    return 0;
}

void View::get_viewplane(double aspect, double zdist,
			 Vector& u, Vector& v)
{
  Vector lookdir(lookat()-eyep());
  Log::log( DEBUG, "[View::get_viewplane] lookat: (" + mkString(lookat().x) + ", " + mkString(lookat().y) + ", " + mkString(lookat().z) + ")" );
  Log::log( DEBUG, "[View::get_viewplane] eyep: (" + mkString(eyep().x) + ", " + mkString(eyep().y) + ", " + mkString(eyep().z) + ")" );
  Log::log( DEBUG, "[View::get_viewplane] lookdir: (" + mkString(lookdir.x) + ", " + mkString(lookdir.y) + ", " + mkString(lookdir.z) + ")" );
  Vector z(lookdir);
  z.normalize();
  Log::log( DEBUG, "[View::get_viewplane] z: (" + mkString(z.x) + ", " + mkString(z.y) + ", " + mkString(z.z) + ")" );
  Log::log( DEBUG, "[View::get_viewplane] up: (" + mkString(up().x) + ", " + mkString(up().y) + ", " + mkString(up().z) + ")" );
  Vector x(Vector::cross(z, up()));
  x.normalize();
  Log::log( DEBUG, "[View::get_viewplane] x: (" + mkString(x.x) + ", " + mkString(x.y) + ", " + mkString(x.z) + ")" );
  Vector y(Vector::cross(x, z));
  Log::log( DEBUG, "[View::get_viewplane] y: (" + mkString(y.x) + ", " + mkString(y.y) + ", " + mkString(y.z) + ")" );
  double xviewsize=zdist*tan(DtoR(fov()/2.))*2.;
  double yviewsize=xviewsize/aspect;
  Log::log( DEBUG, "[View::get_viewplane] zdist = " + mkString(zdist) );
  Log::log( DEBUG, "[View::get_viewplane] xviewsize = " + mkString(xviewsize) );
  Log::log( DEBUG, "[View::get_viewplane] yviewsize = " + mkString(yviewsize) );
  Log::log( DEBUG, "[View::get_viewplane] fov = " + mkString(fov()) );
  Log::log( DEBUG, "[View::get_viewplane] aspect = " + mkString(aspect) );
  x*=xviewsize;
  y*=yviewsize;
  Log::log( DEBUG, "[View::get_viewplane] x * xviewsize: (" + mkString(x.x) + ", " + mkString(x.y) + ", " + mkString(x.z) + ")" );
  Log::log( DEBUG, "[View::get_viewplane] y * yviewsize: (" + mkString(y.x) + ", " + mkString(y.y) + ", " + mkString(y.z) + ")" );
  u=x;
  v=y;
}

void
View::get_normalized_viewplane( Vector& u, Vector& v)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Vector::cross(z, up()));
    x.normalize();
    Vector y(Vector::cross(x, z));
    y.normalize();
    u=x;
    v=y;
}

Point3d View::eyespace_to_objspace(const Point3d& ep, double aspect)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Vector::cross(z, up()));
    x.normalize();
    Vector y(Vector::cross(x, z));
    double xviewsize=tan(DtoR(fov()/2.))*2.;
    double yviewsize=xviewsize/aspect;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;
    x*=xscale;
    y*=yscale;
    
    Point3d p(eyep()+x*ep.x+y*ep.y+z*ep.z);
    return p;
}

Point3d View::objspace_to_eyespace(const Point3d& ep, double aspect)
{
    // first compute basic cordiante frame

    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Vector::cross(z, up()));
    x.normalize();
    Vector y(Vector::cross(x, z));
    double xviewsize=tan(DtoR(fov()/2.))*2.;
    double yviewsize=xviewsize/aspect;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;

    // the transform everything into eyespace
    
//    x*=xscale;
//    y*=yscale;
//    x.x(x.x()/xscale);
//    y.y(y.y()/yscale);
    Vector epv(ep);
    Point3d p(Dot(x,epv-eyep_)/xscale,
	      Dot(y,epv-eyep_)/yscale,
	      Dot(z,epv-eyep_));
    return p;
}
Point3d View::eyespace_to_objspace_ns(const Point3d& ep, double)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Vector::cross(z, up()));
    x.normalize();
    Vector y(Vector::cross(x, z));
    
    Point3d p(eyep()+x*ep.x+y*ep.y+z*ep.z);
    return p;
}

Point3d View::objspace_to_eyespace_ns(const Point3d& ep, double)
{
    // first compute basic cordiante frame

    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Vector::cross(z, up()));
    x.normalize();
    Vector y(Vector::cross(x, z));

    Vector epv(ep);
    Point3d p(Dot(x,epv-eyep_),
	      Dot(y,epv-eyep_),
	      Dot(z,epv-eyep_));
    return p;
}

double View::depth(const Point3d& p)
{
    Vector dir(lookat()-eyep());
    dir.normalize();
    double d=-Dot(eyep(), dir);
    double dist=Dot(p, dir)+d;
    return dist;
}

Point3d View::lookat() const
{
    return lookat_;
}

Point3d View::eyep() const
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

void View::eyep(const Point3d& e)
{
    eyep_=e;
}

void View::lookat(const Point3d& l)
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

}

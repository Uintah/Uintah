/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Geom/View.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/Trig.h>

namespace SCIRun {


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

int
View::operator==(const View& copy)
{
  if ( eyep_ == copy.eyep_ && lookat_ == copy.lookat_ &&
       up_ == copy.up_     && fov_ == copy.fov_           )
    return 1;
  else
    return 0;
}

int
View::operator!=(const View& copy)
{
  if ( eyep_ != copy.eyep_ || lookat_ != copy.lookat_ ||
       up_ != copy.up_     || fov_ != copy.fov_           )
    return 1;
  else
    return 0;
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

void
View::get_normalized_viewplane( Vector& u, Vector& v)
{
    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));
    y.normalize();
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
    Vector epv(ep.vector());
    Point p(Dot(x,epv-eyep_.vector())/xscale,
	    Dot(y,epv-eyep_.vector())/yscale,
	    Dot(z,epv-eyep_.vector()));
    return p;
}
Point View::eyespace_to_objspace_ns(const Point& ep, double)
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

Point View::objspace_to_eyespace_ns(const Point& ep, double)
{
    // first compute basic cordiante frame

    Vector lookdir(lookat()-eyep());
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up()));
    x.normalize();
    Vector y(Cross(x, z));

    Vector epv(ep.vector());
    Point p(Dot(x,epv-eyep_.vector()),
	    Dot(y,epv-eyep_.vector()),
	    Dot(z,epv-eyep_.vector()));
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

ExtendedView::ExtendedView() : View()
{
  xres_ = yres_ = 0;
}

ExtendedView::ExtendedView( const View& v, int x, int y, const Color& c )
: View( v )
{
  xres_ = x;
  yres_ = y;
  bg_   = c;
}

ExtendedView::ExtendedView( const Point& e, const Point& l, const Vector& u,
			   double f, int x, int y, const Color& c )
: View( e, l, u, f )
{
  xres_ = x;
  yres_ = y;
  bg_   = c;
}

ExtendedView::ExtendedView( const ExtendedView& copy )
: View((View)copy), xres_(copy.xres_), yres_(copy.yres_), bg_(copy.bg_)
{
}

Color
ExtendedView::bg() const
{
  return bg_;
}

void
ExtendedView::bg( const Color& c )
{
  bg_ = c;
}

int
ExtendedView::xres() const
{
  return xres_;
}

void
ExtendedView::xres( int x )
{
  xres_ = x;
}

int
ExtendedView::yres() const
{
  return yres_;
}

void
ExtendedView::yres( int y )
{
  yres_ = y;
}

void
ExtendedView::Print( )
{
  printf("raster: ( %i, %i );;; ( %f, %f, %f )", xres(), yres(), bg().r(),
	 bg().g(), bg().b() );
}

#define VIEW_VERSION 1

void Pio(Piostream& stream, View& v)
{
  
    stream.begin_class("View", VIEW_VERSION);
    Pio(stream, v.eyep_);
    Pio(stream, v.lookat_);
    Pio(stream, v.up_);
    Pio(stream, v.fov_);
    stream.end_class();
}

void Pio(Piostream& stream, ExtendedView& v)
{

    stream.begin_class("ExtendedView", VIEW_VERSION);
    Pio(stream, v.eyep_);
    Pio(stream, v.lookat_);
    Pio(stream, v.up_);
    Pio(stream, v.fov_);
    Pio(stream, v.xres_);
    Pio(stream, v.yres_);
    Pio(stream, v.bg_);
    stream.end_class();
}

} // End namespace SCIRun



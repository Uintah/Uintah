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
 *  Cone.h: Cone object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomCone.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>

namespace SCIRun {

using std::cerr;
using std::ostream;

Persistent* make_GeomCone()
{
    return scinew GeomCone;
}

PersistentTypeID GeomCone::type_id("GeomCone", "GeomObj", make_GeomCone);

Persistent* make_GeomCappedCone()
{
    return scinew GeomCappedCone(0,0);
}

PersistentTypeID GeomCappedCone::type_id("GeomCappedCone", "GeomObj", make_GeomCappedCone);

GeomCone::GeomCone(int nu, int nv)
  : GeomObj(), bottom(0,0,0), top(0,0,1), bot_rad(1), top_rad(0), nu(nu), nv(nv)
{
    adjust();
}

GeomCone::GeomCone(const Point& bottom, const Point& top,
		   double bot_rad, double top_rad, int nu, int nv)
: GeomObj(), bottom(bottom), top(top), bot_rad(bot_rad),
top_rad(top_rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomCone::move(const Point& _bottom, const Point& _top,
		    double _bot_rad, double _top_rad, int _nu, int _nv)
{
    bottom=_bottom;
    top=_top;
    bot_rad=_bot_rad;
    top_rad=_top_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomCone::GeomCone(const GeomCone& copy)
: GeomObj(), v1(copy.v1), v2(copy.v2), bottom(copy.bottom), top(copy.top),
bot_rad(copy.bot_rad), top_rad(copy.top_rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomCone::~GeomCone()
{
}

GeomObj* GeomCone::clone()
{
    return scinew GeomCone(*this);
}

void GeomCone::adjust()
{

    axis=top-bottom;
    height=axis.length();
    if(height < 1.e-6){
	cerr << "Degenerate Cone!\n";
    } else {
	axis.find_orthogonal(v1, v2);
    }
    tilt=(bot_rad-top_rad)/axis.length2();
    Vector z(0,0,1);	
    if(Abs(axis.y())+Abs(axis.x()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis)/height;
    zrotangle=-Acos(cangle);
}

void GeomCone::get_bounds(BBox& bb)
{
    bb.extend_disc(bottom, axis, bot_rad);
    bb.extend_disc(top, axis, top_rad);
}

#define GEOMCONE_VERSION 1

void GeomCone::io(Piostream& stream)
{

    stream.begin_class("GeomCone", GEOMCONE_VERSION);
    GeomObj::io(stream);
    Pio(stream, bottom);
    Pio(stream, top);
    Pio(stream, axis);
    Pio(stream, bot_rad);
    Pio(stream, top_rad);
    Pio(stream, nu);
    Pio(stream, nv);
    if(stream.reading())
	adjust();
    stream.end_class();
}

// Capped Geometry

GeomCappedCone::GeomCappedCone(int nu, int nv, int nvdisc1, int nvdisc2)
: GeomCone(nu, nv), nvdisc1(nvdisc1), nvdisc2(nvdisc2)
{
}

GeomCappedCone::GeomCappedCone(const Point& bottom, const Point& top,
			       double bot_rad, double top_rad, int nu, int nv,
			       int nvdisc1, int nvdisc2)
: GeomCone(bottom, top, bot_rad, top_rad, nu, nv), nvdisc1(nvdisc1),
  nvdisc2(nvdisc2)
{
}

GeomCappedCone::GeomCappedCone(const GeomCappedCone& copy)
: GeomCone(copy), nvdisc1(copy.nvdisc1), nvdisc2(copy.nvdisc2)
{
}

GeomCappedCone::~GeomCappedCone()
{
}

GeomObj* GeomCappedCone::clone()
{
    return scinew GeomCappedCone(*this);
}

#define GEOMCAPPEDCONE_VERSION 1

void GeomCappedCone::io(Piostream& stream)
{

    stream.begin_class("GeomCappedCone", GEOMCAPPEDCONE_VERSION);
    GeomCone::io(stream);
    Pio(stream, nvdisc1);
    Pio(stream, nvdisc2);
    stream.end_class();
}


// GeomCones, accelerated for many objects.

Persistent* make_GeomCones()
{
    return new GeomCones();
}

PersistentTypeID GeomCones::type_id("GeomCones", "GeomObj", make_GeomCones);

GeomCones::GeomCones(int nu, double r)
  : radius_(r),
    nu_(nu)
{
}

GeomCones::GeomCones(const GeomCones& copy)
  : radius_(copy.radius_),
    nu_(copy.nu_),
    points_(copy.points_),
    colors_(copy.colors_)
{
}

GeomCones::~GeomCones()
{
}

GeomObj* GeomCones::clone()
{
  return new GeomCones(*this);
}

void
GeomCones::get_bounds(BBox& bb)
{
  for (unsigned int i = 0; i < points_.size(); i+=2)
  {
    Vector axis(points_[i] - points_[i+1]);
    bb.extend_disc(points_[i], axis, radius_);
    bb.extend_disc(points_[i+1], axis, 0);
  }
}

#define GEOMCONES_VERSION 1

void
GeomCones::io(Piostream& stream)
{

  stream.begin_class("GeomCones", GEOMCONES_VERSION);
  GeomObj::io(stream);
  Pio(stream, radius_);
  Pio(stream, nu_);
  Pio(stream, points_);
  Pio(stream, colors_);
  Pio(stream, indices_);
  Pio(stream, radii_);
  stream.end_class();
}


static unsigned char
COLOR_FTOB(double v)
{
  const int inter = (int)(v * 255 + 0.5);
  if (inter > 255) return 255;
  if (inter < 0) return 0;
  return (unsigned char)inter;
}


bool
GeomCones::add(const Point& p1, const Point& p2)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
    return true;
  }
  return false;
}

bool
GeomCones::add(const Point& p1, const Point &p2, const MaterialHandle &c)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
    
    const unsigned char r0 = COLOR_FTOB(c->diffuse.r());
    const unsigned char g0 = COLOR_FTOB(c->diffuse.g());
    const unsigned char b0 = COLOR_FTOB(c->diffuse.b());
    const unsigned char a0 = COLOR_FTOB(c->transparency);

    colors_.push_back(r0);
    colors_.push_back(g0);
    colors_.push_back(b0);
    colors_.push_back(a0);

    return true;
  }
  return false;
}

bool
GeomCones::add(const Point& p1, const Point& p2, float index)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
    indices_.push_back(index);
    return true;
  }
  return false;
}


bool
GeomCones::add_radius(const Point& p1, const Point& p2, double r)
{
  if (add(p1, p2))
  {
    radii_.push_back(r);
    return true;
  }
  return false;
}


bool
GeomCones::add_radius(const Point& p1, const Point &p2,
		      const MaterialHandle &c, double r)
{
  if (add(p1, p2, c))
  {
    radii_.push_back(r);
    return true;
  }
  return false;
}


bool
GeomCones::add_radius(const Point& p1, const Point& p2, float index, double r)
{
  if (add(p1, p2, index))
  {
    radii_.push_back(r);
    return true;
  }
  return false;
}


void
GeomCones::set_nu(int nu)
{
  if (nu < 3) { nu_ = 3; }
  if (nu > 40) { nu_ = 40; }
  else { nu_ = nu; }
}


} // End namespace SCIRun


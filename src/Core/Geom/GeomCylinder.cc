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
 *  Cylinder.h: Cylinder Object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomCylinder.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <iostream>

namespace SCIRun {

using std::cerr;
using std::ostream;
using std::endl;

Persistent* make_GeomCylinder()
{
    return scinew GeomCylinder;
}

PersistentTypeID GeomCylinder::type_id("GeomCylinder", "GeomObj",
				       make_GeomCylinder);

Persistent* make_GeomCappedCylinder()
{
    return scinew GeomCappedCylinder;
}

PersistentTypeID GeomCappedCylinder::type_id("GeomCappedCylinder", "GeomObj",
					     make_GeomCappedCylinder);

GeomCylinder::GeomCylinder(int nu, int nv)
: GeomObj(), bottom(0,0,0), top(0,0,1), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomCylinder::GeomCylinder(const Point& bottom, const Point& top,
			   double rad, int nu, int nv)
: GeomObj(), bottom(bottom), top(top), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomCylinder::move(const Point& _bottom, const Point& _top,
			double _rad, int _nu, int _nv)
{
    bottom=_bottom;
    top=_top;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomCylinder::GeomCylinder(const GeomCylinder& copy)
: GeomObj(copy), v1(copy.v1), v2(copy.v2), bottom(copy.bottom), top(copy.top),
  rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomCylinder::~GeomCylinder()
{
}

void GeomCylinder::adjust()
{
    axis=top-bottom;
    height=axis.length();
    if(height < 1.e-6){
	cerr << "GeomCylinder - Degenerate cylinder!\n";
	cerr << "GeomCylinder - " << top.x() << " " << top.y() << " "   << top.z() << endl;
	cerr << "GeomCylinder - " << bottom.x() << " " << bottom.y() << " "   << bottom.z() << endl;
    } else {
	axis.find_orthogonal(v1, v2);
    }
    v1*=rad;
    v2*=rad;

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

GeomObj* GeomCylinder::clone()
{
    return scinew GeomCylinder(*this);
}

void GeomCylinder::get_bounds(BBox& bb)
{
    bb.extend_cylinder(bottom, axis, rad);
    bb.extend_cylinder(top, axis, rad);
}

#define GEOMCYLINDER_VERSION 1

void GeomCylinder::io(Piostream& stream)
{

    stream.begin_class("GeomCylinder", GEOMCYLINDER_VERSION);
    GeomObj::io(stream);
    Pio(stream, bottom);
    Pio(stream, top);
    Pio(stream, axis);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    if(stream.reading())
	adjust();
    stream.end_class();
}

Persistent* make_GeomColoredCylinders()
{
    return new GeomColoredCylinders();
}

PersistentTypeID GeomColoredCylinders::type_id("GeomColoredCylinders", "GeomObj", make_GeomColoredCylinders);

GeomColoredCylinders::GeomColoredCylinders()
  : radius_(1.0),
    nu_(4),
    nv_(1)
{
}

GeomColoredCylinders::GeomColoredCylinders(const GeomColoredCylinders& copy)
  : radius_(copy.radius_),
    nu_(copy.nu_),
    nv_(copy.nv_),
    points_(copy.points_),
    colors_(copy.colors_)
{
}

GeomColoredCylinders::~GeomColoredCylinders()
{
}

GeomObj* GeomColoredCylinders::clone()
{
  return new GeomColoredCylinders(*this);
}

void
GeomColoredCylinders::get_bounds(BBox& bb)
{
  for (unsigned int i = 0; i < points_.size(); i+=2)
  {
    Vector axis(points_[i] - points_[i+1]);
    bb.extend_cylinder(points_[i], axis, radius_);
    bb.extend_cylinder(points_[i], axis, radius_);
  }
}

#define GEOMLINES_VERSION 1

void
GeomColoredCylinders::io(Piostream& stream)
{

  stream.begin_class("GeomColoredCylinders", GEOMLINES_VERSION);
  GeomObj::io(stream);
  Pio(stream, radius_);
  Pio(stream, nu_);
  Pio(stream, nv_);
  Pio(stream, points_);
  Pio(stream, colors_);
  Pio(stream, indices_);
  stream.end_class();
}


void
GeomColoredCylinders::add(const Point& p1, const Point& p2)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
  }
}

void
GeomColoredCylinders::add(const Point& p1, MaterialHandle c1,
			  const Point& p2, MaterialHandle c2)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
    colors_.push_back(c1);
    colors_.push_back(c2);
  }
}

void
GeomColoredCylinders::add(const Point& p1, float index1,
			  const Point& p2, float index2)
{
  if ((p1 - p2).length2() > 1.0e-12)
  {
    points_.push_back(p1);
    points_.push_back(p2);
    indices_.push_back(index1);
    indices_.push_back(index2);
  }
}

void
GeomColoredCylinders::set_nu_nv(int nu, int nv)
{
  if (nu < 3) { nu_ = 3; }
  if (nu > 20) { nu_ = 20; }
  else { nu_ = nu; }
}


// Capped Geometry....

GeomCappedCylinder::GeomCappedCylinder(int nu, int nv, int nvdisc)
: GeomCylinder(nu, nv), nvdisc(nvdisc)
{
}

GeomCappedCylinder::GeomCappedCylinder(const Point& bottom, const Point& top,
				       double rad, int nu, int nv, int nvdisc)
: GeomCylinder(bottom, top, rad, nu, nv), nvdisc(nvdisc)
{
}

GeomCappedCylinder::GeomCappedCylinder(const GeomCappedCylinder& copy)
: GeomCylinder(copy), nvdisc(copy.nvdisc)
{
}

GeomCappedCylinder::~GeomCappedCylinder()
{
}

GeomObj* GeomCappedCylinder::clone()
{
    return scinew GeomCappedCylinder(*this);
}

#define GEOMCAPPEDCYLINDER_VERSION 1

void GeomCappedCylinder::io(Piostream& stream)
{
    stream.begin_class("GeomCappedCylinder", GEOMCAPPEDCYLINDER_VERSION);
    GeomCylinder::io(stream);
    Pio(stream, nvdisc);
}

} // End namespace SCIRun


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
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

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
    bb.extend_cylinder(bottom, axis, bot_rad);
    bb.extend_cylinder(top, axis, top_rad);
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

} // End namespace SCIRun


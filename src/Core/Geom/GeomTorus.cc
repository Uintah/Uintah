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
 * Torus.cc: Torus objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomTorus.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <iostream>
using std::ostream;

namespace SCIRun {


Persistent* make_GeomTorus()
{
    return scinew GeomTorus;
}

PersistentTypeID GeomTorus::type_id("GeomTorus", "GeomObj", make_GeomTorus);

Persistent* make_GeomTorusArc()
{
    return scinew GeomTorusArc;
}

PersistentTypeID GeomTorusArc::type_id("GeomTorusArc", "GeomTorus", make_GeomTorusArc);

GeomTorus::GeomTorus(int nu, int nv)
: GeomObj(), cen(0,0,0), axis(0,0,1), rad1(1), rad2(.1), nu(nu), nv(nv)
{
    adjust();
}

GeomTorus::GeomTorus(const Point& cen, const Vector& axis,
		     double rad1, double rad2, int nu, int nv)
: GeomObj(), cen(cen), axis(axis), rad1(rad1), rad2(rad2), nu(nu), nv(nv)
{
    adjust();
}

void GeomTorus::move(const Point& _cen, const Vector& _axis,
		     double _rad1, double _rad2, int _nu, int _nv)
{
    cen=_cen;
    axis=_axis;
    rad1=_rad1;
    rad2=_rad2;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomTorus::GeomTorus(const GeomTorus& copy)
: GeomObj(copy), cen(copy.cen), axis(copy.axis),
  rad1(copy.rad1), rad2(copy.rad2), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomTorus::~GeomTorus()
{
}

void GeomTorus::adjust()
{
    axis.normalize();

    Vector z(0,0,1);
    if(Abs(axis.y())+Abs(axis.x()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(axis, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, axis);
    zrotangle=-Acos(cangle);
}

GeomObj* GeomTorus::clone()
{
    return scinew GeomTorus(*this);
}

GeomTorusArc::GeomTorusArc(int nu, int nv)
  : GeomTorus(nu, nv), zero(0,1,0), arc_angle(Pi)
{
}

GeomTorusArc::GeomTorusArc(const Point& cen, const Vector& axis,
			   double rad1, double rad2, const Vector& zero,
			   double start_angle, double arc_angle,
			   int nu, int nv)
  : GeomTorus(cen, axis, rad1, rad2, nu, nv),
    zero(zero), start_angle(start_angle), arc_angle(arc_angle)
{
}

void GeomTorusArc::move(const Point& _cen, const Vector& _axis,
			double _rad1, double _rad2, const Vector& _zero,
			double _start_angle, double _arc_angle, int _nu, int _nv)
{
    cen=_cen;
    axis=_axis;
    rad1=_rad1;
    rad2=_rad2;
    nu=_nu;
    nv=_nv;
    zero=_zero;
    start_angle=_start_angle;
    arc_angle=_arc_angle;
    adjust();
}

GeomTorusArc::GeomTorusArc(const GeomTorusArc& copy)
  : GeomTorus(copy),
    zero(copy.zero),
    start_angle(copy.start_angle),
    arc_angle(copy.arc_angle)
{
}

GeomTorusArc::~GeomTorusArc()
{
}

void GeomTorusArc::adjust()
{
    axis.normalize();
    zero.normalize();
    yaxis=Cross(axis, zero);
}

GeomObj* GeomTorusArc::clone()
{
    return scinew GeomTorusArc(*this);
}

void GeomTorus::get_bounds(BBox& bb)
{
    bb.extend_cylinder(cen-axis*rad2, axis, rad1+rad2);
    bb.extend_cylinder(cen+axis*rad2, axis, rad1+rad2);
}

void GeomTorusArc::get_bounds(BBox& bb)
{
    bb.extend_cylinder(cen-axis*rad2, axis, rad1+rad2);
    bb.extend_cylinder(cen+axis*rad2, axis, rad1+rad2);
}

#define GEOMTORUS_VERSION 1

void GeomTorus::io(Piostream& stream)
{
  
    stream.begin_class("GeomTorus", GEOMTORUS_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, axis);
    Pio(stream, rad1);
    Pio(stream, rad2);
    Pio(stream, nu);
    Pio(stream, nv);
    if(stream.reading())
	adjust();
    stream.end_class();
}

#define GEOMTORUSARC_VERSION 1

void GeomTorusArc::io(Piostream& stream)
{
    stream.begin_class("GeomTorusArc", GEOMTORUSARC_VERSION);
    GeomTorus::io(stream);
    Pio(stream, zero);
    Pio(stream, start_angle);
    Pio(stream, arc_angle);
    Pio(stream, yaxis);
    stream.end_class();
}

} // End namespace SCIRun



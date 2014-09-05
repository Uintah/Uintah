//static char *id="@(#) $Id$";

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

#include <SCICore/Geom/GeomCone.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/Trig.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

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
: GeomObj(), bottom(0,0,0), top(0,0,1), bot_rad(1), top_rad(0),
  nu(nu), nv(nv)
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
  using SCICore::Math::Abs;
  using namespace Geometry;


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
    bb.extend_cyl(bottom, axis, bot_rad);
    bb.extend_cyl(top, axis, top_rad);
}

#define GEOMCONE_VERSION 1

void GeomCone::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomCone", GEOMCONE_VERSION);
    GeomObj::io(stream);
    SCICore::Geometry::Pio(stream, bottom);
    SCICore::Geometry::Pio(stream, top);
    SCICore::Geometry::Pio(stream, axis);
    Pio(stream, bot_rad);
    Pio(stream, top_rad);
    Pio(stream, nu);
    Pio(stream, nv);
    if(stream.reading())
	adjust();
    stream.end_class();
}

bool GeomCone::saveobj(ostream& out, const clString& format,
		       GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->orient(out, bottom+axis*0.5, axis);
	saveinfo->start_node(out, "Cone");
	saveinfo->indent(out);
	out << "parts SIDES\n";
	saveinfo->indent(out);
	out << "bottomRadius " << bot_rad << "\n";
	saveinfo->indent(out);
	out << "height " << height << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->rib_orient(out, bottom, axis);
	saveinfo->indent(out);
	out << "Cone " << height << " " << bot_rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomCone::saveobj");
	return false;
    }
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
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomCappedCone", GEOMCAPPEDCONE_VERSION);
    GeomCone::io(stream);
    Pio(stream, nvdisc1);
    Pio(stream, nvdisc2);
    stream.end_class();
}

bool GeomCappedCone::saveobj(ostream& out, const clString& format,
			   GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv" ){
	NOT_FINISHED("GeomCappedCone::saveobj");
	return false;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->rib_orient(out, bottom, axis);
	saveinfo->indent(out);
	out << "Cone " << height << " " << bot_rad << " 360\n";
	out << "Disk " << 0 << " " << bot_rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomCappedCone::saveobj");
	return false;
    }
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.7  1999/10/07 02:07:41  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/09/04 06:01:47  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.5  1999/08/29 00:46:54  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/28 17:54:39  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:19  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:06  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:38  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:49  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:18  dav
// Import sources
//
//

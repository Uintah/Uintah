//static char *id="@(#) $Id$";

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

#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Math/Trig.h>

namespace SCICore {
namespace GeomSpace {

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
  using SCICore::Math::Abs;
  using namespace Geometry;

    axis=top-bottom;
    height=axis.length();
    if(height < 1.e-6){
	cerr << "Degenerate cylinder!\n";
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
    bb.extend_cyl(bottom, axis, rad);
    bb.extend_cyl(top, axis, rad);
}

#define GEOMCYLINDER_VERSION 1

void GeomCylinder::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

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

bool GeomCylinder::saveobj(ostream& out, const clString& format,
			   GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->orient(out, bottom+axis*0.5, axis);
	saveinfo->start_node(out, "Cylinder");
	saveinfo->indent(out);
	out << "parts SIDES\n";
	saveinfo->indent(out);
	out << "radius " << rad << "\n";
	saveinfo->indent(out);
	out << "height " << height << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->rib_orient(out, bottom, axis);
	saveinfo->indent(out);
	out << "Cylinder " << rad << " 0 " << height << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomCylinder::saveobj");
	return false;
    }
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
    PersistentSpace::Pio(stream, nvdisc);
}

bool GeomCappedCylinder::saveobj(ostream& out, const clString& format,
				 GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->orient(out, bottom+axis*0.5, axis);
	saveinfo->start_node(out, "Cylinder");
	saveinfo->indent(out);
	out << "parts ALL\n";
	saveinfo->indent(out);
	out << "radius " << rad << "\n";
	saveinfo->indent(out);
	out << "height " << height << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->rib_orient(out, bottom, axis);
	saveinfo->indent(out);
	out << "Cylinder " << rad << " 0 " << height << " 360\n";
	out << "Disk " << height << " " << rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomCylinder::saveobj");
	return false;
    }
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:20  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:38  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:49  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

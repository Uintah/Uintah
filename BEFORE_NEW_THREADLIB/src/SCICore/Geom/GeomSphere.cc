//static char *id="@(#) $Id$";

/*
 * GeomSphere.cc: Sphere objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomSphere()
{
    return scinew GeomSphere;
}

PersistentTypeID GeomSphere::type_id("GeomSphere", "GeomObj", make_GeomSphere);

GeomSphere::GeomSphere(int nu, int nv)
: GeomObj(), cen(0,0,0), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomSphere::GeomSphere(const Point& cen, double rad, int nu, int nv)
: GeomObj(), cen(cen), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomSphere::move(const Point& _cen, double _rad, int _nu, int _nv)
{
    cen=_cen;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomSphere::GeomSphere(const GeomSphere& copy)
: GeomObj(copy), cen(copy.cen), rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomSphere::~GeomSphere()
{
}

void GeomSphere::adjust()
{
}

GeomObj* GeomSphere::clone()
{
    return scinew GeomSphere(*this);
}

void GeomSphere::get_bounds(BBox& bb)
{
    bb.extend(cen, rad);
}

#define GEOMSPHERE_VERSION 1

void GeomSphere::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;

    stream.begin_class("GeomSphere", GEOMSPHERE_VERSION);
    GeomObj::io(stream);
    Pio(stream, cen);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

bool GeomSphere::saveobj(ostream& out, const clString& format,
			 GeomSave* saveinfo)
{
    cerr << "saveobj Sphere\n";
    if(format == "vrml" || format == "iv"){
	saveinfo->start_tsep(out);
	saveinfo->start_node(out, "Sphere");
	saveinfo->indent(out);
	out << "radius " << rad << "\n";
	saveinfo->end_node(out);
	saveinfo->end_tsep(out);
	return true;
    } else if(format == "rib"){
	saveinfo->start_trn(out);
	saveinfo->indent(out);
	out << "Translate " << cen.x() << " "  << cen.y() << " "  << cen.z() << "\n";
	saveinfo->indent(out);
	out << "Sphere " << rad << " " << -rad << " " << rad << " 360\n";
	saveinfo->end_trn(out);
	return true;
    } else {
	NOT_FINISHED("GeomSphere::saveobj");
	return false;
    }
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:25  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:52  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

//static char *id="@(#) $Id$";

/*
 *  GeomDisc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomDisc.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/Trig.h>

namespace SCICore {
namespace GeomSpace {

using namespace Geometry;

Persistent* make_GeomDisc()
{
    return scinew GeomDisc;
}

PersistentTypeID GeomDisc::type_id("GeomDisc", "GeomObj", make_GeomDisc);

GeomDisc::GeomDisc(int nu, int nv)
: GeomObj(), n(0,0,1), rad(1), nu(nu), nv(nv)
{
    adjust();
}

GeomDisc::GeomDisc(const Point& cen, const Vector& n,
		   double rad, int nu, int nv)
: GeomObj(), cen(cen), n(n), rad(rad), nu(nu), nv(nv)
{
    adjust();
}

void GeomDisc::move(const Point& _cen, const Vector& _n,
		    double _rad, int _nu, int _nv)
{
    cen=_cen;
    n=_n;
    rad=_rad;
    nu=_nu;
    nv=_nv;
    adjust();
}

GeomDisc::GeomDisc(const GeomDisc& copy)
: GeomObj(), v1(copy.v1), v2(copy.v2), cen(copy.cen), n(copy.n),
  rad(copy.rad), nu(copy.nu), nv(copy.nv)
{
    adjust();
}

GeomDisc::~GeomDisc()
{
}

GeomObj* GeomDisc::clone()
{
    return scinew GeomDisc(*this);
}

void GeomDisc::adjust()
{
  using SCICore::Math::Abs;

    if(n.length2() < 1.e-6){
	cerr << "Degenerate normal on Disc!\n";
    } else {
	n.find_orthogonal(v1, v2);
    }
    n.normalize();
    Vector z(0,0,1);
    if(Abs(n.y()) < 1.e-5){
	// Only in x-z plane...
	zrotaxis=Vector(0,-1,0);
    } else {
	zrotaxis=Cross(n, z);
	zrotaxis.normalize();
    }
    double cangle=Dot(z, n);
    zrotangle=-Acos(cangle);
}

void GeomDisc::get_bounds(BBox& bb)
{
    bb.extend_cyl(cen, n, rad);
}

#define GEOMDISC_VERSION 1

void GeomDisc::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomDisc", GEOMDISC_VERSION);
    GeomObj::io(stream);
    SCICore::Geometry::Pio(stream, cen);
    SCICore::Geometry::Pio(stream, n);
    Pio(stream, rad);
    Pio(stream, nu);
    Pio(stream, nv);
    stream.end_class();
}

bool GeomDisc::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomDisc::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/09/04 06:01:48  sparker
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
// Revision 1.3  1999/08/17 23:50:20  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:39  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:50  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//


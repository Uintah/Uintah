//static char *id="@(#) $Id$";

/*
 *  TriStrip.cc: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomTriStrip.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomTriStrip()
{
    return scinew GeomTriStrip;
}

PersistentTypeID GeomTriStrip::type_id("GeomTriStrip", "GeomObj", make_GeomTriStrip);

Persistent* make_GeomTriStripList()
{
    return scinew GeomTriStripList;
}

PersistentTypeID GeomTriStripList::type_id("GeomTriStripList", "GeomObj",
					   make_GeomTriStripList);

GeomTriStrip::GeomTriStrip()
{
}

GeomTriStrip::GeomTriStrip(const GeomTriStrip& copy)
: GeomVertexPrim(copy)
{
}

GeomTriStrip::~GeomTriStrip() {
}

GeomObj* GeomTriStrip::clone()
{
    return scinew GeomTriStrip(*this);
}

#define GEOMTRISTRIP_VERSION 1

void GeomTriStrip::io(Piostream& stream)
{
    stream.begin_class("GeomTriStrip", GEOMTRISTRIP_VERSION);
    GeomVertexPrim::io(stream);
    stream.end_class();
}

bool GeomTriStrip::saveobj(ostream& out, const clString& format, GeomSave* saveinfo)
{
    if(format == "vrml" || format == "iv") {
      NOT_FINISHED("GeomTriStrip::saveobj");
    } else if(format == "rib") {
	saveinfo->indent(out);
	out << "PointsPolygons " << verts.size() << " [ ";
	int i;
	for(i=0; i < verts.size(); i++)
	  out << "3 ";
	out << "] ";
	for(i=0; i < verts.size(); i++)
	  out << i * 3 << " ";
	out << "] \"P\" [ ";
	for(i=0; i < verts.size(); i++)
	  out << verts[i]->p.x() << " "
	      << verts[i]->p.y() << " "
	      << verts[i]->p.z() << " ";
	out << " ]\n\n";
	cerr << "Should output color and normal here.\n";
    } else {
      NOT_FINISHED("GeomTriStrip::saveobj");
    }
    return false;
}

int GeomTriStrip::size(void)
{
    return verts.size();
}

GeomTriStripList::GeomTriStripList()
:n_strips(0)
{

}

GeomTriStripList::~GeomTriStripList()
{
    // everything should be deleted
}

GeomObj* GeomTriStripList::clone()
{
    return new GeomTriStripList(*this);
}

void GeomTriStripList::add(const Point& p)
{
    int last = pts.size();
    pts.grow(3);
    pts[last++] = float(p.x());
    pts[last++] = float(p.y());
    pts[last++] = float(p.z());

}


void GeomTriStripList::add(const Point& p, const Vector& v)
{
    int last = pts.size();
    pts.grow(3);

    pts[last++] = float(p.x());
    pts[last++] = float(p.y());
    pts[last++] = float(p.z());

    last = nrmls.size();
    nrmls.grow(3);
    nrmls[last++] = float(v.x());
    nrmls[last++] = float(v.y());
    nrmls[last++] = float(v.z());
}

void GeomTriStripList::end_strip(void)
{
    int last = pts.size();
    int last_s = strips.size();
    
    strips.grow(1);
    strips[last_s] = last;
}

int GeomTriStripList::size(void)
{
    return nrmls.size()/3;
}

#define GEOMTRISTRIPLIST_VERSION 1

void GeomTriStripList::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomTriStripList", GEOMTRISTRIPLIST_VERSION);
    GeomObj::io(stream);
    Pio(stream, n_strips);
    SCICore::Containers::Pio(stream, pts);
    SCICore::Containers::Pio(stream, nrmls);
    SCICore::Containers::Pio(stream, strips);
    stream.end_class();
}

bool GeomTriStripList::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomTriStripList::saveobj");
    return false;
}

Point GeomTriStripList::get_pm1(void)
{
    int npts = pts.size()-1;
    Point p(pts[npts-2],pts[npts-1],pts[npts]);
    return p;
}

Point GeomTriStripList::get_pm2(void)
{
    int npts = pts.size()-1-3;
    Point p(pts[npts-2],pts[npts-1],pts[npts]);
    return p;
}

void GeomTriStripList::permute(int i0,int i1, int i2)
{
    float *ptr = &pts[pts.size()-1-8];
    float remap[9];

    remap[0] = ptr[i0*3 + 0];
    remap[1] = ptr[i0*3 + 1];
    remap[2] = ptr[i0*3 + 2];

    remap[3] = ptr[i1*3 + 0];
    remap[4] = ptr[i1*3 + 1];
    remap[5] = ptr[i1*3 + 2];

    remap[6] = ptr[i2*3 + 0];
    remap[7] = ptr[i2*3 + 1];
    remap[8] = ptr[i2*3 + 2];

    for(int i=0;i<9;i++) 
	ptr[i] = remap[i];
}

void GeomTriStripList::get_bounds(BBox& box)
{
    for(int i=0;i<pts.size();i+=3)
	box.extend(Point(pts[i],pts[i+1],pts[i+2]));
}

int GeomTriStripList::num_since(void)
{
    int ssize = strips.size();
    int n_on = pts.size()/3;
    if (ssize) {
	int lastp = strips[ssize-1];
	n_on -= lastp/3;
    }

    return n_on;

}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/10/07 02:07:47  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/29 00:46:56  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/28 17:54:43  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:28  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:16  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:46  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:54  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

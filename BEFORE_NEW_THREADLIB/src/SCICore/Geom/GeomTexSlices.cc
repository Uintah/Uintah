//static char *id="@(#) $Id$";

/*
 *  GeomTexSlices.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Geom/GeomTexSlices.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#ifdef _WIN32
#include <string.h>
#else
#include <strings.h>
#endif

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomTexSlices()
{
    return scinew GeomTexSlices(0,0,0,Point(0,0,0), Point(1,1,1));
}

PersistentTypeID GeomTexSlices::type_id("GeomTexSlices", "GeomObj", make_GeomTexSlices);

GeomTexSlices::GeomTexSlices(int nx, int ny, int nz, const Point& min,
			     const Point &max)
: nx(nx), ny(ny), nz(nz), min(min), max(max), have_drawn(0), accum(0.1),
  bright(0.6)
{
    Xmajor.newsize(nx, ny, nz);
    Ymajor.newsize(ny, nx, nz);
    Zmajor.newsize(nz, nx, ny);
}

GeomTexSlices::GeomTexSlices(const GeomTexSlices& copy)
: GeomObj(copy)
{
}


GeomTexSlices::~GeomTexSlices()
{

}

void GeomTexSlices::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

GeomObj* GeomTexSlices::clone()
{
    return scinew GeomTexSlices(*this);
}

#define GeomTexSlices_VERSION 1

void GeomTexSlices::io(Piostream& stream)
{
    stream.begin_class("GeomTexSlices", GeomTexSlices_VERSION);
    GeomObj::io(stream);
    stream.end_class();
}    

bool GeomTexSlices::saveobj(ostream&, const clString& /*format*/, GeomSave*)
{
  return 0;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/19 23:18:06  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/17 23:50:26  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

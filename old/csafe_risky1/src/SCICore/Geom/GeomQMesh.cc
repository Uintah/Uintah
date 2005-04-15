//static char *id="@(#) $Id$";

/*
 *  GeomQMesh.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Geom/GeomQMesh.h>

#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>
#include <iostream>
using std::ostream;

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomQMesh()
{
    return scinew GeomQMesh(0,0);
}

PersistentTypeID GeomQMesh::type_id("GeomQMesh", "GeomObj", make_GeomQMesh);

GeomQMesh::GeomQMesh(int nr, int nc)
:nrows(nr),ncols(nc)
{
  pts.resize(nr*nc*3);
  nrmls.resize(nr*nc*3);
  clrs.resize(nr*nc);
}

GeomQMesh::GeomQMesh(const GeomQMesh& copy)
: GeomObj(copy)
{
}

GeomQMesh::~GeomQMesh()
{
}

void GeomQMesh::add(int x, int y, Point& p, Vector& v, Color& c)
{
  int index3 = y*nrows*3 + x*3;
  int index  = y*nrows + x;

  pts[index3 + 0] = p.x();
  pts[index3 + 1] = p.y();
  pts[index3 + 2] = p.z();

  nrmls[index3 + 0] = v.x();
  nrmls[index3 + 1] = v.y();
  nrmls[index3 + 2] = v.z();

  clrs[index] = Colorub(c);
}

void GeomQMesh::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size()/3;i++){
	Point pp(pts[i*3 + 0],pts[i*3 + 1],pts[i*3 + 2]);
	bb.extend(pp);
    }
}

GeomObj* GeomQMesh::clone()
{
    return scinew GeomQMesh(*this);
}

#define GeomQMesh_VERSION 2

void GeomQMesh::io(Piostream&)
{

}    

bool GeomQMesh::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomQMesh::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:44  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:24  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:42  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//


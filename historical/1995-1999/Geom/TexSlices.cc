
/*
 *  Grid.cc: Grid object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Geom/TexSlices.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Malloc/Allocator.h>
#include <strings.h>

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

void GeomTexSlices::get_bounds(BSphere& bs)
{
  bs.extend(min);
  bs.extend(max);
}

void GeomTexSlices::make_prims(Array1<GeomObj*>&,
			  Array1<GeomObj*>&)
{
    NOT_FINISHED("GeomTexSlices::make_prims");
}

GeomObj* GeomTexSlices::clone()
{
    return scinew GeomTexSlices(*this);
}

void GeomTexSlices::preprocess()
{
    NOT_FINISHED("GeomTexSlices::preprocess");
}

void GeomTexSlices::intersect(const Ray&, Material*, Hit&)
{
    NOT_FINISHED("GeomTexSlices::intersect");
}

#define GeomTexSlices_VERSION 1

void GeomTexSlices::io(Piostream& stream)
{
    stream.begin_class("GeomTexSlices", GeomTexSlices_VERSION);
    GeomObj::io(stream);
    stream.end_class();
}    

bool GeomTexSlices::saveobj(ostream&, const clString& format, GeomSave*)
{
  return 0;
}


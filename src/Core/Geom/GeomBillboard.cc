
/*
 * Billboard.cc: Pts objects
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Geom/GeomBillboard.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBillboard()
{
    return scinew GeomBillboard(0, Point(0,0,0));
}

PersistentTypeID GeomBillboard::type_id("GeomBillboard", "GeomObj",
					make_GeomBillboard);


GeomBillboard::GeomBillboard(GeomObj* obj, const Point &p)
: child(obj), at(p)
{

}

GeomBillboard::~GeomBillboard()
{
  if(child)
    delete child;
}

GeomObj* GeomBillboard::clone()
{
  cerr << "GeomBillboard::clone not implemented!\n";
  return 0;
}

void GeomBillboard::get_bounds(BBox& box)
{
  //box.extend(Point(-5,-2,-1));
  //box.extend(Point(5,2, 1));
  child->get_bounds(bbox);

  box.reset();
  box.extend( Point( bbox.min().x(), bbox.min().z(), bbox.min().y() ));
  box.extend( Point( bbox.max().x(), bbox.max().z(), bbox.max().y() ));
  box.translate( at.vector() );
  //  cerr << " at " << box.min() << "  " << box.max() << "\n";
}

#define GEOMBBOXCACHE_VERSION 1

void GeomBillboard::io(Piostream& stream)
{

    stream.begin_class("GeomBillboard", GEOMBBOXCACHE_VERSION);
    Pio(stream, child);
    stream.end_class();
}

bool GeomBillboard::saveobj(ostream& out, const clString& format,
			    GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace SCIRun



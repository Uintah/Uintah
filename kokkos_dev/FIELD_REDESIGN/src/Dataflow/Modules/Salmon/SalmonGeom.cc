//static char *id="@(#) $Id$";

/*
 *  SalmonGeom.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <iostream>
using std::cerr;
using std::ostream;

#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geom/BBoxCache.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECommon/Modules/Salmon/SalmonGeom.h>
#include <PSECommon/Modules/Salmon/Roe.h>

namespace PSECommon {
namespace Modules {

using SCICore::PersistentSpace::Persistent;
using SCICore::GeomSpace::GeomBBoxCache;
using SCICore::Thread::CrowdMonitor;

Persistent* make_GeomSalmonItem()
{
    return scinew GeomSalmonItem;
}

PersistentTypeID GeomSalmonItem::type_id("GeomSalmonItem", "GeomObj",
					 make_GeomSalmonItem);

GeomSalmonPort::GeomSalmonPort(int no)
:portno(no),msg_head(0),msg_tail(0)
{
    // just use default constructor for base class...
}

GeomSalmonPort::~GeomSalmonPort()
{
    // maybee flush mesages, or do nothing...
}

GeomSalmonItem::GeomSalmonItem()
:child(0),lock(0)
{
}

GeomSalmonItem::GeomSalmonItem(GeomObj* obj,const clString& nm, 
			       CrowdMonitor* lck)
:child(obj),name(nm),lock(lck)
{
    if (!lock)
	child = new GeomBBoxCache(obj);
}

GeomSalmonItem::~GeomSalmonItem()
{
    if (child)
	delete child;  // I'm not sure if this should be here...
}

void GeomSalmonItem::get_triangles( Array1<float> &v)
{
  if ( child )
    child->get_triangles(v);
}

GeomObj* GeomSalmonItem::clone()
{
    cerr << "GeomSalmonItem::clone not implemented!\n";
    return 0;
}

void GeomSalmonItem::reset_bbox()
{
    child->reset_bbox();
}

void GeomSalmonItem::get_bounds(BBox& box)
{
    child->get_bounds(box);
}

#define GEOMSALMONITEM_VERSION 1

void GeomSalmonItem::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::GeomSpace::Pio;

    stream.begin_class("GeomSalmonItem", GEOMSALMONITEM_VERSION);
    int have_lock;
    if(stream.writing())
	have_lock=lock?1:0;
    Pio(stream, have_lock);
    if(stream.reading())
	if(have_lock)
	    lock=new CrowdMonitor("GeomSalmonItem crowd monitor");
	else
	    lock=0;
    Pio(stream, name);
    Pio(stream, child);
    stream.end_class();
}

bool GeomSalmonItem::saveobj(ostream& out, const clString& format,
			     GeomSave* saveinfo)
{
    return child->saveobj(out, format, saveinfo);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5.2.2  2000/10/26 10:03:41  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5.2.1  2000/09/28 03:16:07  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.6  2000/06/06 15:08:17  dahart
// - Split OpenGL.cc into OpenGL.cc and OpenGL.h to allow class
// derivations of the OpenGL renderer.
// - Added a constructor to the Salmon class with a Module name parameter
// to allow derivations of Salmon with different names.
// - Added get_triangles() to SalmonGeom for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.5  1999/10/07 02:06:57  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/29 00:46:43  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.3  1999/08/17 23:50:16  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:37:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:53  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//

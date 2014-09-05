//static char *id="@(#) $Id$";

/*
 *  Switch.cc:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */
#include <SCICore/Geom/Switch.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

Persistent* make_GeomSwitch()
{
    return scinew GeomSwitch(0,0);
}

PersistentTypeID GeomSwitch::type_id("GeomSwitch", "GeomObj", make_GeomSwitch);

Persistent* make_GeomTimeSwitch()
{
    return scinew GeomTimeSwitch(0,0,0);
}

PersistentTypeID GeomTimeSwitch::type_id("GeomTimeSwitch", "GeomObj", make_GeomTimeSwitch);

GeomSwitch::GeomSwitch(GeomObj* obj, int state)
: GeomContainer(obj), state(state)
{
}

GeomSwitch::GeomSwitch(const GeomSwitch& copy)
: GeomContainer(copy), state(copy.state)
{
}

GeomSwitch::~GeomSwitch()
{
}

GeomObj* GeomSwitch::clone()
{
    return scinew GeomSwitch(*this);
}

void GeomSwitch::set_state(int st)
{
   state=st;
}

int GeomSwitch::get_state()
{
   return state;
}

void GeomSwitch::get_bounds(BBox& bbox)
{
   if(state)child->get_bounds(bbox);
}


bool GeomSwitch::saveobj(ostream& out, const clString& format,
			 GeomSave* saveinfo)
{
    cerr << "saveobj Switch ";
    if(state)
      {
	cerr << "yep.\n";
	return child->saveobj(out, format, saveinfo);
      }
    else
      {
	cerr << "nope.\n";
	return true;
      }
}

#define GEOMSWITCH_VERSION 1

void GeomSwitch::io(Piostream& stream)
{
    stream.begin_class("GeomSwitch", GEOMSWITCH_VERSION);
    GeomContainer::io(stream);
    PersistentSpace::Pio(stream, state);
    stream.end_class();
}

GeomTimeSwitch::GeomTimeSwitch(GeomObj* obj, double tbeg, double tend)
: GeomContainer(obj), tbeg(tbeg), tend(tend)
{
}

GeomTimeSwitch::GeomTimeSwitch(const GeomTimeSwitch& copy)
: GeomContainer(copy), tbeg(copy.tbeg), tend(copy.tend)
{
}

GeomTimeSwitch::~GeomTimeSwitch()
{
}

GeomObj* GeomTimeSwitch::clone()
{
    return scinew GeomTimeSwitch(*this);
}

#define GEOMTIMESWITCH_VERSION 1

void GeomTimeSwitch::io(Piostream& stream)
{
    stream.begin_class("GeomSwitch", GEOMTIMESWITCH_VERSION);
    GeomContainer::io(stream);
    PersistentSpace::Pio(stream, tbeg);
    PersistentSpace::Pio(stream, tend);
    stream.end_class();
}

bool GeomTimeSwitch::saveobj(ostream&, const clString&, GeomSave*)
{
    NOT_FINISHED("GeomTimeSwitch::saveobj");
    return false;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:50  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:33  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:23  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:52  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

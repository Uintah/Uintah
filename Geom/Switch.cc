
/*
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */
#include <Geom/Switch.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>

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


void GeomSwitch::get_bounds(BSphere& bs)
{
   if(state)child->get_bounds(bs);
}

void GeomSwitch::make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree)
{
   if(state)child->make_prims(free, dontfree);
}


void GeomSwitch::preprocess()
{
   if(state)child->preprocess();
}


void GeomSwitch::intersect(const Ray& ray, Material* matl, Hit& hit)
{
   if(state)child->intersect(ray, matl, hit);
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
    Pio(stream, state);
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
    Pio(stream, tbeg);
    Pio(stream, tend);
    stream.end_class();
}

bool GeomTimeSwitch::saveobj(ostream&, const clString& format, GeomSave*)
{
    NOT_FINISHED("GeomTimeSwitch::saveobj");
    return false;
}


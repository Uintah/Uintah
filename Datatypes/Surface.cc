/*
 *  Surface.cc: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Surface.h>
#include <Classlib/NotFinished.h>
#include <Geometry/Grid.h>

PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep)
: rep(rep), grid(0)
{
}

Surface::~Surface()
{
    destroy_grid();
}

Surface::Surface(const Surface& copy)
{
    NOT_FINISHED("Surface::Surface");
}

void Surface::destroy_grid()
{
    if (grid) delete grid;
}

#define SURFACE_VERSION 2

void Surface::io(Piostream& stream) {
    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    if (version >= 2) {
	Pio(stream, conductivity);
#ifdef __GNUG__
	int bt=bdry_type;
	Pio(stream, bt);
	bdry_type=bt;
#else
	Pio(stream, bdry_type);
#endif
    }
    stream.end_class();
}

TriSurface* Surface::getTriSurface()
{
    if(rep==TriSurf)
	return (TriSurface*)this;
    else
	return 0;
}

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

PersistentTypeID Surface::type_id("Surface", "Datatype", 0);

Surface::Surface(Representation rep)
: rep(rep)
{
}

Surface::~Surface() {
}

Surface::Surface(const Surface& copy)
{
    NOT_FINISHED("Surface::Surface");
}

#define SURFACE_VERSION 1

void Surface::io(Piostream& stream) {
    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    stream.end_class();
}

TriSurface* Surface::getTriSurface()
{
    if(rep==TriSurf)
	return (TriSurface*)this;
    else
	return 0;
}

    

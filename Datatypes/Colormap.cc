
/*
 *  Colormap.h: Colormap definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Colormap.h>
#include <Classlib/String.h>

PersistentTypeID Colormap::type_id("Colormap", "Datatype", 0);

Colormap::Colormap(int nlevels)
: colors(nlevels)
{
}

Colormap::~Colormap()
{
}

Colormap* Colormap::clone()
{
    return 0;
}

#define COLORMAP_VERSION 1

void Colormap::io(Piostream& stream)
{
    int version=stream.begin_class("Colormap", COLORMAP_VERSION);
    Pio(stream, colors);
    stream.end_class();
}

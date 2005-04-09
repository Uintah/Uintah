
/*
 *  Lighting.cc:  The light sources in a scene
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Lighting.h>
#include <Geom/Light.h>
#include <Classlib/Persistent.h>
#include <Classlib/String.h>
#include <iostream.h>

Lighting::Lighting()
: amblight(Color(0,0,0))
{
}

Lighting::~Lighting()
{
}

#define LIGHTING_VERSION 1

void Pio(Piostream& stream, Lighting& l)
{
    stream.begin_class("Lighting", LIGHTING_VERSION);
    Pio(stream, l.lights);
    Pio(stream, l.amblight);
    stream.end_class();
}


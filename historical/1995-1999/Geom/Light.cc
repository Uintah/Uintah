
/*
 *  Light.cc: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Light.h>

PersistentTypeID Light::type_id("Light", "Persistent", 0);

Light::Light(const clString& name)
: name(name)
{
}

Light::~Light()
{
}

void Pio(Piostream& stream, Light*& light)
{
    Persistent* tlight=light;
    stream.io(tlight, Light::type_id);
    if(stream.reading())
	light=(Light*)tlight;
}

#define LIGHT_VERSION 1

void Light::io(Piostream& stream)
{
    stream.begin_class("Light", LIGHT_VERSION);
    Pio(stream, name);
    stream.end_class();
}

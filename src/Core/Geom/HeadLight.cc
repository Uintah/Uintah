
/*
 *  HeadLight.cc:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/HeadLight.h>
#include <Core/Geom/View.h>

namespace SCIRun {

Persistent* make_HeadLight()
{
    return new HeadLight("", Color(0,0,0));
}

PersistentTypeID HeadLight::type_id("HeadLight", "Light", make_HeadLight);

HeadLight::HeadLight(const clString& name, const Color& c)
: Light(name), c(c)
{
}

HeadLight::~HeadLight()
{
}

#define HEADLIGHT_VERSION 1

void HeadLight::io(Piostream& stream)
{

    stream.begin_class("HeadLight", HEADLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, c);
    stream.end_class();
}


} // End namespace SCIRun


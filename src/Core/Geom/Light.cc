//static char *id="@(#) $Id$";

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

namespace SCICore {
namespace GeomSpace {

PersistentTypeID Light::type_id("Light", "Persistent", 0);

Light::Light(const clString& name)
: name(name)
{
}

Light::~Light()
{
}

#define LIGHT_VERSION 1

void Light::io(Piostream& stream)
{
    stream.begin_class("Light", LIGHT_VERSION);
    PersistentSpace::Pio(stream, name);
    stream.end_class();
}

} // End namespace GeomSpace


namespace PersistentSpace {

using namespace GeomSpace;

void Pio(Piostream& stream, Light*& light)
{
    Persistent* tlight=light;
    stream.io(tlight, Light::type_id);
    if(stream.reading())
	light=(Light*)tlight;
}

} // End namespace PersistentSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//


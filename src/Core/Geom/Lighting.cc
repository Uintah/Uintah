//static char *id="@(#) $Id$";

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
#include <Persistent/Persistent.h>
#include <Containers/String.h>
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

Lighting::Lighting()
: amblight(Color(0,0,0))
{
}

Lighting::~Lighting()
{
}

} // End namespace GeomSpace


namespace PersistentSpace {

#define LIGHTING_VERSION 1

void Pio( Piostream& stream, GeomSpace::Lighting & l )
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("Lighting", LIGHTING_VERSION);
    Pio(stream, l.lights);
    Pio(stream, l.amblight);
    stream.end_class();
}

} // End namespace PersistentSpace



} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:55  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

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

#include <SCICore/Geom/Lighting.h>
#include <SCICore/Geom/Light.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>
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

#define LIGHTING_VERSION 1

void Pio( Piostream& stream, Lighting & l )
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("Lighting", LIGHTING_VERSION);
    Pio(stream, l.lights);
    GeomSpace::Pio(stream, l.amblight);
    stream.end_class();
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:19  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

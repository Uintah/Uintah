//static char *id="@(#) $Id$";

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

#include <SCICore/Geom/HeadLight.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Util/NotFinished.h>

namespace SCICore {
namespace GeomSpace {

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
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("HeadLight", HEADLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    GeomSpace::Pio(stream, c);
    stream.end_class();
}


} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:30  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:18  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:48  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

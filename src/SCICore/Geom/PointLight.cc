//static char *id="@(#) $Id$";

/*
 *  PointLight.cc:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/PointLight.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace GeomSpace {

Persistent* make_PointLight()
{
    return new PointLight("", Point(0,0,0), Color(0,0,0));
}

PersistentTypeID PointLight::type_id("PointLight", "Light", make_PointLight);

PointLight::PointLight(const clString& name,
		       const Point& p, const Color& c)
: Light(name), p(p), c(c)
{
}

PointLight::~PointLight()
{
}

#define POINTLIGHT_VERSION 1

void PointLight::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    using SCICore::GeomSpace::Pio;

    stream.begin_class("PointLight", POINTLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, p);
    Pio(stream, c);
    stream.end_class();
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/17 23:50:32  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:21  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:50  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:56  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

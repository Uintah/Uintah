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
#include <SCICore/Geom/GeomRaytracer.h>
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

void HeadLight::lintens(const OcclusionData& od, const Point& hit_position,
			 Color& light, Vector& light_dir)
{
    NOT_FINISHED("HeadLight::lintens");
#if 0
    if(od.level == 0){
	// No need to do intersection test - we won't hit anything.
	light=c;
	light_dir=od.view->eyep()-p;
	light_dir.normalize();
    } else {
	light_dir=od.view->eyep()-p;
	double light_dist=light_dir.normalize();
	double atten=od.raytracer->light_ray(p, od.view->eyep(), light_dir, light_dist);
	light=c*atten;
    }
#endif
}

void HeadLight::compute_lighting(const View& view, const Point& at,
				  Color& color, Vector& to)
{
    to=at-view.eyep();
    to.normalize();
    color=c;
}

GeomObj* HeadLight::geom()
{
    return 0; // Never seen
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

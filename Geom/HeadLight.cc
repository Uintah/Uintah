
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

#include <Geom/HeadLight.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/View.h>
#include <Classlib/NotFinished.h>

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

void HeadLight::lintens(const OcclusionData&, const Point&,
			Color&, Vector&)
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

#define HEADLIGHT_VERSION 1

void HeadLight::io(Piostream& stream)
{
    stream.begin_class("HeadLight", HEADLIGHT_VERSION);
    // Do the base class first...
    Light::io(stream);
    Pio(stream, c);
    stream.end_class();
}

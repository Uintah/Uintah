
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

void HeadLight::lintens(const OcclusionData& od, const Point& p,
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

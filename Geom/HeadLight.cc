
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
#include <Classlib/NotFinished.h>
#include <Geom/GeomRaytracer.h>
#include <Geom/View.h>

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
    to=at-view.eyep;
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
    if(od.level == 0){
	// No need to do intersection test - we won't hit anything.
	light=c;
	light_dir=od.view->eyep-p;
	light_dir.normalize();
    } else {
	NOT_FINISHED("HeadLight::lintens");
	light=Color(1,1,1);
	light_dir=Vector(0,1,0);
    }
}


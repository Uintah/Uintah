
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

#include <Geom/PointLight.h>
#include <Classlib/NotFinished.h>
#include <Geom/Sphere.h>
#include <Malloc/Allocator.h>

PointLight::PointLight(const clString& name,
		       const Point& p, const Color& c)
: Light(name), p(p), c(c)
{
}

PointLight::~PointLight()
{
}

void PointLight::compute_lighting(const View&, const Point& at,
				  Color& color, Vector& to)
{
    to=at-p;
    to.normalize();
    color=c;
}

GeomObj* PointLight::geom()
{
    return scinew GeomSphere(p, 1.0);
}

void PointLight::lintens(const OcclusionData&, const Point&,
			 Color& light, Vector& light_dir)
{
    NOT_FINISHED("PointLight::lintens");
    light=Color(1,1,1);
    light_dir=Vector(0,1,0);
}


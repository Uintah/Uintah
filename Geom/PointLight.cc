
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

PointLight::PointLight(const Point& p, const Color& c)
: p(p), c(c)
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

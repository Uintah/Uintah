
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
#include <Geom/View.h>

HeadLight::HeadLight(const Color& c)
: c(c)
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


/*
 *  View.cc:  The camera
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/View.h>

View::View()
{
}

View::~View()
{
}

View::View(const Point& eyep, const Point& lookat, const Vector& up,
	   double fov)
: eyep(eyep), lookat(lookat), up(up), fov(fov)
{
}

View::View(const View& copy)
: eyep(copy.eyep), lookat(copy.lookat), up(copy.up), fov(copy.fov)
{
}

View& View::operator=(const View& copy)
{
    eyep=copy.eyep;
    lookat=copy.lookat;
    up=copy.up;
    fov=copy.fov;
    return *this;
}


/*
 *  PointLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_PointLight_h
#define SCI_Geom_PointLight_h 1

#include <Geom/Light.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>

class PointLight : public Light {
    Point p;
    Color c;
public:
    PointLight(const Point&, const Color&);
    virtual ~PointLight();
    virtual void compute_lighting(const Point& at, Color&, Vector&);
};

#endif /* SCI_Geom_PointLight_h */


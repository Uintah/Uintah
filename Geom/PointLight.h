
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
    PointLight(const clString& name, const Point&, const Color&);
    virtual ~PointLight();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&);
    virtual GeomObj* geom();
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
};

#endif /* SCI_Geom_PointLight_h */


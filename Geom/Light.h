
/*
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <variant.h>

class Color;
class DrawInfoOpenGL;
class Point;
class Vector;
class View;

class Light {
public:
    Light();
    virtual ~Light();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&)=0;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
};

#endif /* SCI_Geom_Light_h */


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

class Color;
class Point;
class Vector;

class Light {
public:
    Light();
    virtual ~Light();
    virtual void compute_lighting(const Point& at, Color&, Vector&)=0;
};

#endif /* SCI_Geom_Light_h */

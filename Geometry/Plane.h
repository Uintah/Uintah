/*
 *  Plane.h: Directed plane
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Plane_h
#define SCI_project_Plane_h 1

#include <Geometry/Point.h>

class Plane {
protected:
    double a,b,c,d;
public:
    Plane(const Plane &copy);
    Plane(const Point &p1, const Point &p2, const Point &p3);
    Plane(double a, double b, double c, double d);
    ~Plane();
    double eval_point(const Point &p);
    void flip();
};

#endif

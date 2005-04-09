
/*
 *  Plane.cc: Uniform Plane containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geometry/Plane.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

Plane::Plane(double a, double b, double c, double d)
: a(a), b(b), c(c), d(d) {
}

Plane::Plane(const Plane &copy)
: a(copy.a), b(copy.b), c(copy.c), d(copy.d) {
}

Plane::Plane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    Vector n(Cross(v2,v1));
    a=n.x(); b=n.y(); c=n.z();
    d=-(p1.x()*a+p1.y()*b+p1.z()*c);
}

Plane::~Plane() {
}

void Plane::flip() {
    a=-a; b=-b; c=-c; c=-d;
}

double Plane::eval_point(const Point &p) {
    return (p.x()*a+p.y()*b+p.z()*c+d);
}


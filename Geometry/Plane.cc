
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

Plane::Plane(const Plane &copy)
: n(copy.n), d(copy.d)
{
}

Plane::Plane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}

Plane::~Plane() {
}

void Plane::flip() {
   n.x(-n.x());
   n.y(-n.y());
   n.z(-n.z());
}

double Plane::eval_point(const Point &p) {
    return Dot(p, n)+d;
}

Point Plane::project(const Point& p)
{
   return p-n*(d+Dot(p,n));
}

Vector Plane::normal()
{
   return n;
}

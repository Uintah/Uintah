
/*
 *  Ray.h:  The Ray datatype
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_Geometry_Ray_h
#define sci_Geometry_Ray_h

#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class Ray {
    Point o;
    Vector d;
public:
    Ray(const Point&, const Vector&);
    Ray(const Ray&);
    ~Ray();
    Ray& operator=(const Ray&);

    Point origin() const;
    Vector direction() const;

    void direction(const Vector& newdir);
};

#endif


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

#ifndef Geometry_Ray_h
#define Geometry_Ray_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {


class Ray {
    Point o;
    Vector d;
public:
    Ray(){}
    Ray(const Point&, const Vector&);
    
    Ray(const Ray&);
    ~Ray();
    Ray& operator=(const Ray&);

    Point origin() const;
    Vector direction() const;
  Point parameter(double t) const; // returns the Point at parameter t
			     //  does not pre-normalized direction
  
  void normalize(); // normalizes the direction vector
    void direction(const Vector& newdir);
};


} // End namespace SCIRun
#endif

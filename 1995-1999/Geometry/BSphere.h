
/*
 *  BSphere.h: Bounding Sphere's
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef sci_Geometry_BSphere_h
#define sci_Geometry_BSphere_h 1

#include <Geometry/Point.h>
class Vector;
class Piostream;
class Ray;

class BSphere {
protected:
friend void Pio(Piostream &, BSphere& );
    int have_some;
    Point cen;
    double rad;
    double rad2;
public:
    BSphere();
    ~BSphere();
    BSphere(const BSphere&);
    inline int valid(){return have_some;}
    void reset();
    void extend(const Point& p);
    void extend(const Point& p, double radius);
    void extend(const BSphere& b);
    Point center() const;
    double radius() const;
    double volume();
    int intersect(const Ray& ray);
};

#endif


/*
 *  BBox.h: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <share/share.h>

#include <Geometry/Point.h>
#include <Geometry/Plane.h>

namespace SCICore {

namespace Geometry {
  class BBox;
}

namespace PersistentSpace {
  class Piostream;
}

namespace Geometry {

#define EEpsilon  1e-13

class Vector;

class SHARE BBox {

protected:
    friend void Pio( Piostream &, BBox& );

    int have_some;
    Point cmin;
    Point cmax;
    Point bcmin, bcmax;
    Point extracmin;
    int inbx, inby, inbz;

public:
    BBox();
    ~BBox();
    BBox(const BBox&);
    inline int valid() const {return have_some;}
    void reset();
    void extend(const Point& p);
    void extend(const Point& p, double radius);
    void extend(const BBox& b);
    void extend_cyl(const Point& cen, const Vector& normal, double r);
    Point center() const;
    double longest_edge();
    void translate(const Vector &v);
    void scale(double s, const Vector &o);
    Point min() const;
    Point max() const;
    Vector diagonal() const;

    inline int inside(const Point &p) const {return (have_some && p.x()>=cmin.x() && p.y()>=cmin.y() && p.z()>=cmin.z() && p.x()<=cmax.x() && p.y()<=cmax.y() && p.z()<=cmax.z());}

    // prepares for intersection by assigning the closest bbox corner
    // to extracmin and initializing an epsilon bigger bbox
    
    void PrepareIntersect( const Point& e );
    
    // returns true if the ray hit the bbox and returns the hit point
    // in hitNear

    int Intersect( const Point& e, const Vector& v, Point& hitNear );

    // given a t distance, assigns the hit point.  returns true
    // if the hit point lies on a bbox face

    int TestTx( const Point& e, const Vector& v, double tx, Point& hitNear );
    int TestTy( const Point& e, const Vector& v, double ty, Point& hitNear );
    int TestTz( const Point& e, const Vector& v, double tz, Point& hitNear );
};

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:55  mcq
// Initial commit
//
// Revision 1.5  1999/07/09 00:27:38  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.4  1999/07/07 21:10:58  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:56:15  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:16  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif

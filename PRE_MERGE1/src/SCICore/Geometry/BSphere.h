
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

#include <share/share.h>

#include <Geometry/Point.h>

namespace SCICore {
namespace Geometry {

class Ray;
class Vector;

class SHARE BSphere {
protected:

    friend void Pio( Piostream &, BSphere& );

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

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:55  mcq
// Initial commit
//
// Revision 1.5  1999/07/09 00:27:39  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.4  1999/07/07 21:10:59  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:56:15  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:17  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif


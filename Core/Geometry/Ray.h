
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

#include <SCICore/share/share.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Geometry {

class SCICORESHARE Ray {
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

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:57  mcq
// Initial commit
//
// Revision 1.4  1999/07/09 00:27:39  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.3  1999/05/06 19:56:16  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:18  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif

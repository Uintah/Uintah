
/*
 *  ContourSet.h: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ContourSet_h
#define SCI_project_ContourSet_h 1

#include <CoreDatatypes/Datatype.h>

#include <Containers/Array1.h>
#include <Containers/LockingHandle.h>
#include <Containers/String.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Transform;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ContourSet;
typedef LockingHandle<ContourSet> ContourSetHandle;

class ContourSet : public Datatype {
public:
    Array1<Array1<Point> > contours;
    Array1<double> conductivity;
    int bdry_type;
    Vector basis[3];
    Vector origin;
    BBox bbox;
    double space;
    clString name;

    ContourSet();
    ContourSet(const ContourSet &copy);
    virtual ~ContourSet();
    virtual ContourSet* clone();
    void translate(const Vector &v);
    void scale(double sc);
    void rotate(const Vector &rot);
    void build_bbox();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:20  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:46  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:37  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:26  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:35  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_ContourSet_h */

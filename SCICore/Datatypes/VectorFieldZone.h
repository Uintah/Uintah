
/*
 *  VectorFieldZone.h: A compound Vector field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldZone_h
#define SCI_project_VectorFieldZone_h 1

#include <CoreDatatypes/VectorField.h>
#include <Containers/Array1.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::Array1;

class VectorFieldZone : public VectorField {
public:
    Array1<VectorFieldHandle> zones;
    VectorFieldZone(int nzones);
    virtual ~VectorFieldZone();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int& cache, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:00  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:51  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:22  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif

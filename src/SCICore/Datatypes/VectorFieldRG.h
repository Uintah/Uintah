
/*
 *  VectorFieldRG.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldRG_h
#define SCI_project_VectorFieldRG_h 1

#include <CoreDatatypes/VectorField.h>
#include <Containers/Array1.h>
#include <Containers/Array3.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::Array1;
using SCICore::Containers::Array3;

class VectorFieldRG : public VectorField {
public:
    int nx;
    int ny;
    int nz;
    Array3<Vector> grid;
    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);

    void resize(int, int, int);
    void set_bounds(const Point&, const Point&);

    VectorFieldRG();
    virtual ~VectorFieldRG();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:32  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:59  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:50  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:21  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif

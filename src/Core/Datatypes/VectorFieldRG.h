
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

#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::Containers::Array3;

class SCICORESHARE VectorFieldRG : public VectorField {
public:
    int nx;
    int ny;
    int nz;
    Array3<Vector> grid;
    virtual Point get_point(int, int, int);
    virtual void locate(const Point&, int&, int&, int&);

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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/12/28 20:45:18  kuzimmer
// added cell-centered data structures
//
// Revision 1.3  1999/08/25 03:48:46  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:59  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif

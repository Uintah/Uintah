
/*
 *  VectorFieldUG.h: Vector Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldUG_h
#define SCI_project_VectorFieldUG_h 1

#include <CoreDatatypes/VectorField.h>

#include <Containers/Array1.h>
#include <CoreDatatypes/Mesh.h>

namespace SCICore {
namespace CoreDatatypes {

class VectorFieldUG : public VectorField {
public:
    MeshHandle mesh;
    Array1<Vector> data;

    enum Type {
	NodalValues,
	ElementValues
    };
    Type typ;

    VectorFieldUG(Type typ);
    VectorFieldUG(const MeshHandle&, Type typ);
    virtual ~VectorFieldUG();
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
// Revision 1.1  1999/07/27 16:56:33  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:59  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:50  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:31  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:48  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif

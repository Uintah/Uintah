
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorField_h
#define SCI_project_VectorField_h 1

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class VectorFieldRG;
class VectorFieldUG;
class VectorField;
class VectorFieldOcean;
typedef LockingHandle<VectorField> VectorFieldHandle;

class SCICORESHARE VectorField : public Datatype {
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;
    virtual void compute_bounds()=0;

protected:
    enum Representation {
	RegularGrid,
	UnstructuredGrid,
	OceanFile,
	Zones
    };
    VectorField(Representation);
private:
    Representation rep;
public:
    virtual ~VectorField();
    virtual VectorField* clone()=0;

    VectorFieldRG* getRG();
    VectorFieldUG* getUG();
    VectorFieldOcean* getOcean();
    void get_bounds(Point&, Point&);
    double longest_dimension();
    virtual int interpolate(const Point&, Vector&)=0;
    virtual int interpolate(const Point&, Vector&, int& cache, int exhaustive=0)=0;
    virtual void get_boundary_lines(Array1<Point>& lines)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:58  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:49  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:20  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_VectorField_h */

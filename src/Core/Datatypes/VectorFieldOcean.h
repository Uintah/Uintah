
/*
 *  VectorFieldOcean.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldOcean_h
#define SCI_project_VectorFieldOcean_h 1

#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomObj;
  }
}

namespace SCICore {
namespace Datatypes {

using SCICore::GeomSpace::GeomObj;
using SCICore::Containers::clString;

class SCICORESHARE VectorFieldOcean : public VectorField {
public:
    clString filename;
    float* data;
    int nx;
    int ny;
    int nz;
    int* depth;
    Array1<double> depthval;
    void locate(const Point&, int&, int&, int&);

    VectorFieldOcean(const clString& filename, const clString& depthfilename);
    virtual ~VectorFieldOcean();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    GeomObj* makesurf(int downsample);
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:45  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:58  sparker
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
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif

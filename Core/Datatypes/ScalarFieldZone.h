
/*
 *  ScalarFieldZone.h: A compound scalar field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldZone_h
#define SCI_project_ScalarFieldZone_h 1

#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/Containers/Array1.h>

namespace SCICore {
namespace CoreDatatypes {

class SCICORESHARE ScalarFieldZone : public ScalarField {
public:
    Array1<ScalarFieldHandle> zones;
    ScalarFieldZone(int nzones);
    virtual ~ScalarFieldZone();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:54  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:28  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:55  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:46  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:17  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

#endif

/*
 *  VectorFieldHUG.h: Vector Fields defined on a hexahedral grid
 *
 *  Written by:
 *   Peter A. Jensen
 *   Sourced from VectorFieldHUG.h
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*******************************************************************************
* Version control
*******************************************************************************/

#ifndef SCI_project_VectorFieldHUG_h
#define SCI_project_VectorFieldHUG_h 1


/*******************************************************************************
* Includes
*******************************************************************************/

#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Datatypes/HexMesh.h>
#include <SCICore/Containers/Array1.h>


/*******************************************************************************
* Hexahedral unstructured grid class
*******************************************************************************/

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

class SCICORESHARE VectorFieldHUG : public VectorField
{
  public:
  
    HexMesh * mesh;
    Array1<Vector> data;
  
    VectorFieldHUG();
    VectorFieldHUG(HexMesh * m);
    virtual ~VectorFieldHUG();
    virtual VectorField* clone();

    virtual void compute_bounds();
    
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int& ix, int exh=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    
    static PersistentTypeID type_id;
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
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:58  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:49  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:31  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:47  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif

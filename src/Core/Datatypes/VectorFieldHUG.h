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

#include <CoreDatatypes/VectorField.h>
#include <CoreDatatypes/HexMesh.h>
#include <Containers/Array1.h>


/*******************************************************************************
* Hexahedral unstructured grid class
*******************************************************************************/

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

class VectorFieldHUG : public VectorField
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

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
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
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:47  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif


/*
 *  ScalarFieldHUG.h: Scalar Fields defined on a hexahedral grid
 *
 *  Written by:
 *   Peter A. Jensen
 *   Sourced from ScalarFieldHUG.h
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*******************************************************************************
* Version control
*******************************************************************************/

#ifndef SCI_project_ScalarFieldHUG_h
#define SCI_project_ScalarFieldHUG_h 1


/*******************************************************************************
* Includes
*******************************************************************************/

#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/CoreDatatypes/HexMesh.h>
#include <SCICore/Containers/Array1.h>


/*******************************************************************************
* Hexahedral unstructured grid class
*******************************************************************************/

namespace SCICore {
namespace CoreDatatypes {

class SCICORESHARE ScalarFieldHUG : public ScalarField 
{
  public:
  
    HexMesh * mesh;
    Array1<double> data;
  
    ScalarFieldHUG();
    ScalarFieldHUG(HexMesh * m);
    virtual ~ScalarFieldHUG();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:49  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:24  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:49  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:42  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:28  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:39  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif

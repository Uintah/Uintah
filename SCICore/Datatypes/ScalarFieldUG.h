
/*
 *  ScalarFieldUG.h: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldUG_h
#define SCI_project_ScalarFieldUG_h 1

#include <CoreDatatypes/ScalarField.h>
#include <CoreDatatypes/Mesh.h>
#include <Containers/Array1.h>

namespace SCICore {
namespace CoreDatatypes {

class ScalarFieldUG : public ScalarField {
public:
    MeshHandle mesh;
    Array1<double> data;

    enum Type {
	NodalValues,
	ElementValues
    };
    Type typ;

    ScalarFieldUG(Type typ);
    ScalarFieldUG(const MeshHandle&, Type typ);
    virtual ~ScalarFieldUG();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void compute_samples(int);  // for random distributions in fields
    virtual void distribute_samples();

    // this has to be called before 2 functions below...

    virtual void fill_gradmags();

    // diferent ways to augment a mesh...
    // 1/grad, grad, histogram of gradients

    // this is just special cased because of potential missing elements
    // you might want to get rid of it and use the base class...

    virtual void over_grad_augment(double vol_wt, double grad_wt, 
				   double crit_scale);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:28  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:55  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:46  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:29  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:43  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif

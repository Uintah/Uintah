
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

#include <Datatypes/ScalarField.h>
#include <Datatypes/Mesh.h>
#include <Classlib/Array1.h>

class ScalarFieldUG : public ScalarField {
public:
    MeshHandle mesh;
    Array1<double> data;

    ScalarFieldUG();
    ScalarFieldUG(const MeshHandle&);
    virtual ~ScalarFieldUG();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

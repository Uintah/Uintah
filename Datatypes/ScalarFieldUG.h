
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
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6);
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

#endif

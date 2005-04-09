
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

#include <VectorField.h>
#include <Mesh.h>
#include <Classlib/Array1.h>

class VectorFieldUG : public VectorField {
public:
    MeshHandle mesh;
    Array1<Vector> data;

    VectorFieldUG();
    VectorFieldUG(const MeshHandle&);
    virtual ~VectorFieldUG();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);

    virtual void io(Piostream&);
    static PersistentTypeID typeid;
};

#endif


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

#include <Core/Datatypes/VectorField.h>

#include <Core/Containers/Array1.h>
#include <Core/Datatypes/Mesh.h>

namespace SCIRun {

class SCICORESHARE VectorFieldUG : public VectorField {
public:
    MeshHandle mesh;
    Array1<Vector> data;

    enum Type {
	NodalValues,
	ElementValues
    };
    Type typ;

    VectorFieldUG(Type typ);
    VectorFieldUG(const MeshHandle&, Type typ);
    virtual ~VectorFieldUG();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif

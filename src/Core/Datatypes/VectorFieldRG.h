
/*
 *  VectorFieldRG.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldRG_h
#define SCI_project_VectorFieldRG_h 1

#include <Core/Datatypes/VectorField.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>

namespace SCIRun {


class SCICORESHARE VectorFieldRG : public VectorField {
public:
    int nx;
    int ny;
    int nz;
    Array3<Vector> grid;
    virtual Point get_point(int, int, int);
    bool locate(int *loc, const Point &p);

    void set_bounds(const Point&, const Point&);

    VectorFieldRG(int x, int y, int z);
    virtual ~VectorFieldRG();
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

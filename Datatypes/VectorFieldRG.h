
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

#include <Datatypes/VectorField.h>
#include <Classlib/Array3.h>

class VectorFieldRG : public VectorField {
public:
    int nx;
    int ny;
    int nz;
    Array3<Vector> grid;
    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);

    void resize(int, int, int);
    void set_minmax(const Point&, const Point&);

    VectorFieldRG();
    virtual ~VectorFieldRG();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

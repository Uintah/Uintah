
/*
 *  VectorFieldZone.h: A compound Vector field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldZone_h
#define SCI_project_VectorFieldZone_h 1

#include <Datatypes/VectorField.h>
#include <Classlib/Array1.h>

class VectorFieldZone : public VectorField {
public:
    Array1<VectorFieldHandle> zones;
    VectorFieldZone(int nzones);
    virtual ~VectorFieldZone();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int& cache);
    virtual void get_boundary_lines(Array1<Point>& lines);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

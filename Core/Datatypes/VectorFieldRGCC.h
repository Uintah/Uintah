
/*
 *  VectorFieldRGCC.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldRGCC_h
#define SCI_project_VectorFieldRGCC_h 1

#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>

namespace SCIRun {


class SCICORESHARE VectorFieldRGCC : public VectorFieldRG {
public:
    Point get_point(int, int, int);
    virtual bool locate(int *loc, const Point &p);

    VectorFieldRGCC();
    virtual ~VectorFieldRGCC();
    virtual VectorField* clone();

    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif

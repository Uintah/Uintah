
/*
 *  ScalarFieldRG.h: Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldRG_h
#define SCI_project_ScalarFieldRG_h 1

#include <ScalarField.h>
#include <Classlib/Array3.h>

class ScalarFieldRG : public ScalarField {
public:
    int nx;
    int ny;
    int nz;
    Array3<double> grid;
    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);

    void resize(int, int, int);

    ScalarFieldRG();
    virtual ~ScalarFieldRG();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&);

    virtual void io(Piostream&);
    static PersistentTypeID typeid;
};

#endif


/*
 *  ScalarFieldZone.h: A compound scalar field type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldZone_h
#define SCI_project_ScalarFieldZone_h 1

#include <Datatypes/ScalarField.h>
#include <Classlib/Array1.h>

class ScalarFieldZone : public ScalarField {
public:
    Array1<ScalarFieldHandle> zones;
    ScalarFieldZone(int nzones);
    virtual ~ScalarFieldZone();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual void get_boundary_lines(Array1<Point>& lines);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

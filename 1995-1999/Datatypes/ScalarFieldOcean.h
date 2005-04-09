
/*
 *  ScalarFieldOcean.h: float Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldOcean_h
#define SCI_project_ScalarFieldOcean_h 1

#include <Datatypes/ScalarFieldRGBase.h>

class ScalarFieldOcean : public ScalarFieldRGBase {
public:
    clString filename;
    float* data;
    int nx, ny, nz;

    ScalarFieldOcean(const clString& filename);
    ScalarFieldOcean(const ScalarFieldOcean&);
    virtual ~ScalarFieldOcean();
    virtual ScalarField* clone();

    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int&, double epsilon1=1.e-6, double epsilon2=1.e-6);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif


/*
 *  VectorFieldOcean.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldOcean_h
#define SCI_project_VectorFieldOcean_h 1

#include <Datatypes/VectorField.h>
#include <Classlib/String.h>
#include <Classlib/Array1.h>
class GeomObj;

class VectorFieldOcean : public VectorField {
public:
    clString filename;
    float* data;
    int nx;
    int ny;
    int nz;
    int* depth;
    Array1<double> depthval;
    void locate(const Point&, int&, int&, int&);

    VectorFieldOcean(const clString& filename, const clString& depthfilename);
    virtual ~VectorFieldOcean();
    virtual VectorField* clone();

    virtual void compute_bounds();
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    GeomObj* makesurf(int downsample);
};

#endif

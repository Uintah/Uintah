
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarField_h
#define SCI_project_ScalarField_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

class ScalarFieldRG;
class ScalarFieldUG;
class ScalarFieldRGchar;
class ScalarField;
typedef LockingHandle<ScalarField> ScalarFieldHandle;

class ScalarField : public Datatype {
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;
    virtual void compute_bounds()=0;

    int have_minmax;
    double data_min;
    double data_max;
    virtual void compute_minmax()=0;
protected:
    enum Representation {
	RegularGrid,
	RegularGridChar,
	UnstructuredGrid,
    };
    ScalarField(Representation);
private:
    Representation rep;
public:
    virtual ~ScalarField();
    virtual ScalarField* clone()=0;

    ScalarFieldRG* getRG();
    ScalarFieldUG* getUG();
    ScalarFieldRGchar* getRGChar();
    void get_minmax(double&, double&);
    void get_bounds(Point&, Point&);
    double longest_dimension();
    virtual Vector gradient(const Point&)=0;
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6)=0;
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_ScalarField_h */


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

#include <Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

class ScalarFieldRG;
class ScalarFieldUG;
class ScalarField;
typedef LockingHandle<ScalarField> ScalarFieldHandle;

class ScalarField : public Datatype {
protected:
#if 0
    int have_bounds;
    Point min;
    Point max;
    Vector diagonal;
    virtual void compute_bounds();
#endif
protected:
    enum Representation {
	RegularGrid,
	TetraHedra,
    };
    ScalarField(Representation);
private:
    Representation rep;
public:
    virtual ~ScalarField();

    ScalarFieldRG* getRG();
    ScalarFieldUG* getUG();
    void get_minmax(double&, double&);
    double longest_dimension();
    virtual Vector gradient(const Point&);

    // Persistent representation...
    virtual void io(Piostream&);
};

#endif /* SCI_project_ScalarField_h */

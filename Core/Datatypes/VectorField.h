
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorField_h
#define SCI_project_VectorField_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {


class VectorFieldRG;
class VectorFieldUG;
class VectorField;
class VectorFieldOcean;
typedef LockingHandle<VectorField> VectorFieldHandle;

class SCICORESHARE VectorField : public Datatype {
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;
    virtual void compute_bounds()=0;

protected:
    enum Representation {
	RegularGrid,
	UnstructuredGrid,
	OceanFile,
	Zones
    };
    VectorField(Representation);
private:
    Representation rep;
public:
    virtual ~VectorField();
    virtual VectorField* clone()=0;

    VectorFieldRG* getRG();
    VectorFieldUG* getUG();
    VectorFieldOcean* getOcean();
    void get_bounds(Point&, Point&);
    double longest_dimension();
    virtual int interpolate(const Point&, Vector&)=0;
    virtual int interpolate(const Point&, Vector&, int& cache, int exhaustive=0)=0;
    virtual void get_boundary_lines(Array1<Point>& lines)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_project_VectorField_h */


/*
 *  Field3.h: The Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Field3D_h
#define SCI_project_Field3D_h 1

#include <Persistent.h>
#include <Classlib/Array3.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>
#include <Multitask/ITC.h>

class Field3D;
typedef LockingHandle<Field3D> Field3DHandle;

class Field3D : public Persistent {
protected:
    friend class LockingHandle<Field3D>;
    int ref_cnt;
    Mutex lock;
private:
    Point min;
    Point max;
    Vector diagonal;
    int nx, ny, nz;
    Array3<double> s_grid;
    Array3<Vector> v_grid;
    int ntetra;
public:
    Field3D();
    ~Field3D();
    enum Representation {
	RegularGrid,
	TetraHedra,
    };
    enum FieldType {
	ScalarField,
	VectorField,
    };
private:
    Representation rep;
    FieldType fieldtype;
public:

    Representation get_rep();
    FieldType get_type();

    // These methods work for all types of representations
    int interpolate(const Point&, Vector&);
    int interpolate(const Point&, double&);
    void get_minmax(double&, double&);

    // Only for regular grids
    int get_nx();
    int get_ny();
    int get_nz();
    void get_n(int&, int&, int&);
    Vector*** get_dataptr();
    double get(int, int, int);
    Point get_point(int, int, int);
    void locate(const Point&, int&, int&, int&);

    // Only for tetrahedra
    int get_ntetra();

    // Persistent representation...
    virtual void io(Piostream&);

    // For writing grids...
    void set_type(FieldType fieldtype);
    void set_rep(Representation rep);
    void set_size(int, int, int);
    void set(int, int, int, const Vector&);
    void set(int, int, int, double);
};

#endif /* SCI_project_Field3DPort_h */


/*
 *  Field3DPort.h: Handle to the Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Field3DPort_h
#define SCI_project_Field3DPort_h 1

#include <Persistent.h>
#include <Port.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

class Field3D : public Persistent {
protected:
    friend class Field3DHandle;
    int ref_cnt;
private:
    int nx, ny, nz;
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
public:

    Representation get_rep();

    // These methods work for all types of representations
    Vector interp_vector(Point&);
    double interp_scalar(Point&);

    // Only for regular grids
    int get_nx();
    int get_ny();
    int get_nz();
    void get_n(int&, int&, int&);

    // Only for tetrahedra
    int get_ntetra();
};

class Field3DHandle {
    Field3D* rep;
public:
    Field3DHandle();
    Field3DHandle(Field3D*);
    Field3DHandle(const Field3DHandle&);
    Field3DHandle& operator=(const Field3DHandle&);
    Field3DHandle& operator=(Field3D*);
    ~Field3DHandle();
    Field3D* operator->() const;
    Field3D* get_rep() const;
};

class Field3DIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
    };

protected:
    friend class Field3DOPort;
public:
    Field3DIPort(Module*, const clString& name, int protocol);
    virtual ~Field3DIPort();
    virtual void reset();
    virtual void finish();

    Field3DHandle get_field();
};

class Field3DOPort : public OPort {
    Field3DIPort* in;
public:
    Field3DOPort(Module*, const clString& name, int protocol);
    virtual ~Field3DOPort();

    virtual void reset();
    virtual void finish();
};

#endif /* SCI_project_Field3DPort_h */


/*
 *  Surface.h: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Surface_h
#define SCI_project_Surface_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>

class Point;
class TriSurface;
class Grid;
class Surface;
typedef LockingHandle<Surface> SurfaceHandle;

class Surface : public Datatype {
protected:
    enum Representation {
	TriSurf,
    };
    Surface(Representation);
private:
    Representation rep;
public:
    clString name;
    Grid *grid;
    enum Boundary_type {
	Interior,
	Exterior,
	VSource,
	ISource,
    };
    Boundary_type bdry_type;
    
    Array1<double> conductivity;	// this will hold the conductivity
                                  // tensor of the stuff this surface encloses.
    Surface(const Surface& copy);
    virtual ~Surface();
    virtual Surface* clone()=0;
    virtual int inside(const Point& p)=0;
    virtual void construct_grid(int, int, int, const Point &, double)=0;
    virtual void destroy_grid();
    TriSurface* getTriSurface();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_Surface_h */


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
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>

class TriSurface;
class Grid;
class Surface;
typedef LockingHandle<Surface> SurfaceHandle;

class Surface : public Datatype {
protected:
    enum Representation {
	TriSurf,
	Other,
    };
    Surface(Representation, int closed);
private:
    Representation rep;
public:
    int hash_x;
    int hash_y;
    Point hash_min;
    double resolution;
    int closed;
    clString name;
    Grid *grid;
    HashTable<int, int> *pntHash;
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
    virtual void construct_grid()=0;
    virtual void destroy_grid();
    virtual void destroy_hash();
    TriSurface* getTriSurface();

    virtual void get_surfpoints(Array1<Point>&)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_Surface_h */

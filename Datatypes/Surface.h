
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
#include <Classlib/LockingHandle.h>

class Surface;
typedef LockingHandle<Surface> SurfaceHandle;

#include <Datatypes/ColorMap.h>
#include <Datatypes/Mesh.h>
#include <Classlib/Array1.h>
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>

class ScalarTriSurface;
class TriSurface;
class PointsSurface;
class Grid;
class SurfTree;
class Surface : public Datatype {
protected:
    enum Representation {
	TriSurf,
	PointsSurf,
	ScalarTriSurf,
	STree,
	Other
    };
    Surface(Representation, int closed);
private:
    Representation rep;
public:
    CrowdMonitor monitor;
    int hash_x;
    int hash_y;
    Point hash_min;
    double resolution;
    int closed;
    clString name;
    Grid *grid;
    HashTable<int, int> *pntHash;

    // Boundary conditions...
    enum BoundaryType {
	DirichletExpression,
	DirichletData,
	None
    };
    clString boundary_expr;
    BoundaryType boundary_type;
    void set_bc(const clString& expr);

    Surface(const Surface& copy);
    virtual ~Surface();
    virtual Surface* clone()=0;
    virtual int inside(const Point& p)=0;
    virtual void construct_grid(int, int, int, const Point &, double)=0;
    virtual void construct_grid()=0;
    virtual void destroy_grid();
    virtual void destroy_hash();
    SurfTree* getSurfTree();
    ScalarTriSurface* getScalarTriSurface();
    TriSurface* getTriSurface();
    PointsSurface* getPointsSurface();
    virtual void get_surfnodes(Array1<NodeHandle>&)=0;
    virtual GeomObj* get_obj(const ColorMapHandle&)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_Surface_h */

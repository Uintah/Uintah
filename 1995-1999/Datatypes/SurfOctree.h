
/*
 *  SurfOctree.h: SurfOctrees
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_SurfOctree_h
#define SCI_project_SurfOctree_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>

class SurfOctreeTop;
typedef LockingHandle<SurfOctreeTop> SurfOctreeTopHandle;

#include <Classlib/Array1.h>
#include <Classlib/Array3.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/VectorFieldRG.h>
#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

#include <stdlib.h>	// for size_t

class SurfOctree {
public:
    int nx;
    int ny;
    int nz;

    char edges;
    Array1<int> matl;
    SurfOctree* child[2][2][2];
    SurfOctree();
    SurfOctree(int nx, int ny, int nz, const Point &min, const Point &max,
	       ScalarFieldRGint* sf, int, int, int);

    SurfOctree(const SurfOctree&);
    ~SurfOctree();
    void print(int x, int y, int z);
    Array1<int>* propagate_up_materials();
    void* operator new(size_t);
    void operator delete(void*, size_t);
    friend void Pio(Piostream&, SurfOctree*&);
};



class SurfOctreeTop;
typedef LockingHandle<SurfOctreeTop> SurfOctreeTopHandle;

class SurfOctreeTop : public Datatype {
public:
    SurfOctree* tree;
    int nx;
    int ny;
    int nz;
    Point min;
    Point max;
    Vector dv;

    SurfOctreeTop();
    SurfOctreeTop(ScalarFieldRGint*);
    SurfOctreeTop(const SurfOctreeTop&);
    virtual SurfOctreeTop* clone();
    virtual ~SurfOctreeTop();
    void print();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

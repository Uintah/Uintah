
/*
 *  Octree.h: Octrees
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_Octree_h
#define SCI_project_Octree_h 1

#include <Classlib/Array1.h>
#include <Classlib/Array3.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/Datatype.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/VectorFieldRG.h>
#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

#include <stdlib.h>	// for size_t

class OctreeTop;
class Octree {
public:
    int nx;
    int ny;
    int nz;

    int bits;
    int leaf;
    int last_leaf;
    int trunk;

    Vector avg_vec;
    double min_sc;
    double max_sc;
    double avg_sc;
    Point mid;

    Octree* child[2][2][2];
    double corner_s[2][2][2];
    Vector corner_v[2][2][2];
    Point corner_p[2][2][2];

    Octree();
    Octree(int nx, int ny, int nz, const Point &min, const Point &max);
    Octree(const Octree&);
    ~Octree();
    void* operator new(size_t);
    void operator delete(void*, size_t);

    void prune();
    Octree *which_child(const Point&);
    Octree *index_child(int bb);
    void insert_scalar_field(ScalarFieldRG* sf);
    void insert_vector_field(VectorFieldRG* vf);
    double set_and_return_max_scalar();
    double set_and_return_min_scalar();
    double set_and_return_avg_scalar();
    Vector set_and_return_avg_vector();
    double set_and_return_corner_scalar(int i, int j, int k);
    Vector set_and_return_corner_vector(int i, int j, int k);
    void build_scalar_tree();
    void build_vector_tree();
    int push_level(const Point&);
    int pop_level(const Point&);
    void push_all_levels();
    void pop_all_levels();
    void top_level();
    void bottom_level();

    void print_tree(int level);
    double get_scalar(int x, int y, int z);
    Vector get_vector(int x, int y, int z);
    Point get_midpoint(int x, int y, int z);

    double get_scalar(const Point &p);
    Vector get_vector(const Point &p);
    int set_scalar(int x, int y, int z, double val);
    int set_vector(int x, int y, int z, const Vector &v);
    int set_scalar(const Point &p, double val);
    int set_vector(const Point &p, const Vector &v);
    void erase_last_level_scalars();
    void erase_last_level_vectors();
    friend void Pio(Piostream&, Octree*&);
};



class OctreeTop;
typedef LockingHandle<OctreeTop> OctreeTopHandle;

class OctreeTop : public Datatype {
public:
    Octree* tree;
    int nx;
    int ny;
    int nz;
    int vectors;
    int scalars;
    int tensors;

    OctreeTop();
    OctreeTop(int nx, int ny, int nz, const BBox& b);
    OctreeTop(const OctreeTop&);
    virtual OctreeTop* clone();
    virtual ~OctreeTop();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif

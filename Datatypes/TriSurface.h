
/*
 *  TriSurface.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_TriSurface_h
#define SCI_Datatypes_TriSurface_h 1

#include <Datatypes/Surface.h>

#include <Classlib/Array1.h>
#include <Geometry/Point.h>
#include <stdlib.h> // For size_t

struct TSElement {
    int i1; 
    int i2; 
    int i3;
    inline TSElement(int i1, int i2, int i3):i1(i1), i2(i2), i3(i3){}
    inline TSElement(const TSElement& e):i1(e.i1), i2(e.i2), i3(e.i3){}
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

void Pio(Piostream&, TSElement*&);

class TriSurface : public Surface {
public:
    Array1<Point> points;
    Array1<TSElement*> elements;
private:
    int empty_index;
    int directed;	// are the triangle all ordered clockwise?
    double distance(const Point &p, int i, int *type, Point *pp=0);
    int find_or_add(const Point &p);
    void add_node(Array1<NodeHandle>& nodes,
		  char* id, const Point& p, int n);
public:
    TriSurface(Representation r=TriSurf);
    TriSurface(const TriSurface& copy, Representation r=TriSurf);
    virtual ~TriSurface();
    virtual Surface* clone();

    // pass in allocated surfaces for conn and d_conn. NOTE: contents will be
    // overwritten
    void separate(int idx, TriSurface* conn, TriSurface* d_conn);

    // NOTE: if elements have been added or removed from the surface
    // remove_empty_index() MUST be called before passing a TriSurface
    // to another module!  
    void remove_empty_index();
    void order_faces();
    inline int is_directed() {return directed;}
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual int inside(const Point& p);
    virtual void construct_hash(int, int, const Point &, double);
    void add_point(const Point& p);
    int add_triangle(int i1, int i2, int i3, int cw=0);
    void remove_triangle(int i);
    double distance(const Point &p, Array1<int> &res, Point *pp=0);

    // these two were implemented for isosurfacing btwn two surfaces
    // (MorphMesher3d module/class)
    int cautious_add_triangle(const Point &p1,const Point &p2,const Point &p3,
			      int cw=0);
    int get_closest_vertex_id(const Point &p1,const Point &p2,
			      const Point &p3);

    virtual GeomObj* get_obj(const ColorMapHandle&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Datatytpes_TriSurface_h */

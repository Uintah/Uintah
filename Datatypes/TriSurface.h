
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

struct TSElement {
    int i1; 
    int i2; 
    int i3;
    inline TSElement(int i1, int i2, int i3):i1(i1), i2(i2), i3(i3){}
};

void Pio(Piostream&, TSElement*&);

class TriSurface : public Surface {
public:
    Array1<Point> points;
    Array1<TSElement*> elements;
private:
    int empty_index;
    int directed;	// are the triangle all ordered clockwise?
    double distance(const Point &p, int i, int *type);
    int find_or_add(const Point &p);
public:
    TriSurface();
    TriSurface(const TriSurface& copy);
    virtual ~TriSurface();
    virtual Surface* clone();

    // NOTE: if elements have been added or removed from the surface
    // remove_empty_index() MUST be called before passing a TriSurface
    // to another module!  
    void remove_empty_index();
    void order_faces();
    inline int is_directed() {return directed;}
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual int inside(const Point& p);
    void add_point(const Point& p);
    int add_triangle(int i1, int i2, int i3);
    int add_triangle(int i1, int i2, int i3, int cw);
    void remove_triangle(int i);
    double distance(const Point &p, Array1<int> &res);

    // these two were implemented for isosurfacing btwn two surfaces
    // (MorphMesher3d module/class)
    int cautious_add_triangle(const Point &p1,const Point &p2,const Point &p3);
    int get_closest_vertex_id(const Point &p1,const Point &p2,
			      const Point &p3);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Datatytpes_TriSurface_h */


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
    int ordered_cw;	// are the triangle all ordered clockwise?
public:
    TriSurface();
    TriSurface(const TriSurface& copy);
    virtual ~TriSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    void add_point(const Point& p);
    int add_triangle(int i1, int i2, int i3);
    void remove_triangle(int i);
    double distance(const Point &p, int i, int *type);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Datatytpes_TriSurface_h */

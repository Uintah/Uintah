
/*
 *  Group.h:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Group_h
#define SCI_Geom_Group_h 1

#include <Geom/Geom.h>
#include <Geometry/BBox.h>

class GeomGroup : public GeomObj {
    Array1<GeomObj*> objs;
    BBox bb;
public:
    GeomGroup();
    GeomGroup(const GeomGroup&);
    virtual ~GeomGroup();
    virtual GeomObj* clone();

    void add(GeomObj*);
    int size();

    virtual void get_bounds(BBox&);

    virtual void objdraw(DrawInfoOpenGL*);
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
};

#endif /* SCI_Geom_Group_h */

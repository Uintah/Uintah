
/*
 *  Geom.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GeomOpenGL_h
#define SCI_Geom_GeomOpenGL_h 1

#include <Classlib/Stack.h>
class Material;

struct DrawInfoOpenGL {
    DrawInfoOpenGL();

    int polycount;
    enum DrawType {
	WireFrame,
	Flat,
	Gouraud,
	Phong,
    };
    DrawType drawtype;

    int lighting;
    int currently_lit;
    int pickmode;

    Material* current_matl;
    Stack<Material*> stack;
    void set_matl(Material*);
    void push_matl(Material*);
    void pop_matl();
};

#endif /* SCI_Geom_GeomOpenGL_h */


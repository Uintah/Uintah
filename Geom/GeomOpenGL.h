
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
#include <GL/glu.h>
class Material;

struct DrawInfoOpenGL {
    DrawInfoOpenGL();
    ~DrawInfoOpenGL();

    int polycount;
    enum DrawType {
	WireFrame,
	Flat,
	Gouraud,
	Phong,
    };
private:
    DrawType drawtype;
public:
    void set_drawtype(DrawType dt);
    inline DrawType get_drawtype() {return drawtype;}

    int lighting;
    int currently_lit;
    int pickmode;

    Material* current_matl;
    Material* appl_matl;
    Material** stack;
    int sp;
    void set_matl();
    void set_matl(Material*);
    void push_matl(Material*);
    void pop_matl();

    GLUquadricObj* qobj;
};

#endif /* SCI_Geom_GeomOpenGL_h */



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

#include <GL/glu.h>
class Material;
class Roe;

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

    void init_lighting(int use_light);
    int lighting;
    int currently_lit;
    int pickmode;
    int fog;

    Material* current_matl;
    void set_matl(Material*);

    int ignore_matl;

    GLUquadricObj* qobj;

    Roe* roe;
    int debug;
    void reset();
};

#endif /* SCI_Geom_GeomOpenGL_h */


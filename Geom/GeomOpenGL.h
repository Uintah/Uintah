
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

const int CLIP_P0 = 1;
const int CLIP_P1 = 2;
const int CLIP_P2 = 4;
const int CLIP_P3 = 8;
const int CLIP_P4 = 16;
const int CLIP_P5 = 32;

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
    void init_clip(void);
    int lighting;
    int currently_lit;
    int pickmode;
    int fog;

    int check_clip; // see if you should ignore clipping planes
    
    int clip_planes; // clipping planes that are on
    double point_size; // so points and lines can be thicker than 1 pixel

    Material* current_matl;
    void set_matl(Material*);

    int ignore_matl;

    GLUquadricObj* qobj;

    Roe* roe;
    int debug;
    void reset();
};

#endif /* SCI_Geom_GeomOpenGL_h */


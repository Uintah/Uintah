
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
#include <GL/glx.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>
class Material;
class Roe;

const int CLIP_P0 = 1;
const int CLIP_P1 = 2;
const int CLIP_P2 = 4;
const int CLIP_P3 = 8;
const int CLIP_P4 = 16;
const int CLIP_P5 = 32;

const int MULTI_TRANSP_FIRST_PASS=2; // 1 is just if you are doing mpasses...

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

    Vector view;  // view vector...
    int axis;     // which axis you are working with...
    int dir;      // direction +/- -> depends on the view...

    double abs_val; // value wi/respect view
    double axis_val; // value wi/respect axis -> pt for comparison...

    double axis_delt; // delta wi/respect axis

    int multiple_transp; // if you have multiple transparent objects...

    void init_view(double znear, double zfar, Point& eyep, Point& lookat);

    Roe* roe;
    Display *dpy;
    int debug;
    void reset();
};

#endif /* SCI_Geom_GeomOpenGL_h */


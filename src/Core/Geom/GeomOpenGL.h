/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

  
/*
 *  GeomOpenGL.h: Displayable Geometry
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

#ifdef _WIN32
#define WINGDIAPI __declspec(dllimport)
#define APIENTRY __stdcall
#define CALLBACK APIENTRY
#endif

#include <stddef.h>
#include <stdlib.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <sci_config.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {


#define GEOM_FONT_COUNT 5

class ViewWindow;
class Material;

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
	Gouraud
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
    int pickchild;
    int npicks;
    int fog;
    int cull;
    int dl;

    int check_clip; // see if you should ignore clipping planes
    
    int clip_planes; // clipping planes that are on

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

    ViewWindow* viewwindow;
  
    double ambient_scale_;
    double diffuse_scale_;
    double specular_scale_;
    double emission_scale_;
    double shininess_scale_;
    double point_size_; // so points can be thicker than 1 pixel
    double line_width_; // so lines can be thicker than 1 pixel
    double polygon_offset_factor_; // so lines and points are offset from faces
    double polygon_offset_units_; // so lines and points are offset from faces

#ifndef _WIN32
    Display *dpy;
#endif
    int debug;
    void reset();

    // Font support.
    bool init_font(int a);

    int    fontstatus[GEOM_FONT_COUNT];
    GLuint fontbase[GEOM_FONT_COUNT];

    bool using_cmtexture_;
    GLuint cmtexture_;  // GeomColorMap texture, 1D
};

} // End namespace SCIRun


#endif /* SCI_Geom_GeomOpenGL_h */


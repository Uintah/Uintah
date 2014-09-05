/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <stddef.h>
#include <stdlib.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <map>
#include <list>

namespace SCIRun {


#define GEOM_FONT_COUNT 5

class ViewWindow;
class Material;
class GeomDL;

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
    void set_material(Material*);

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

    int using_cmtexture_;
    GLuint cmtexture_;  // GeomColorMap texture, 1D

    std::map<GeomDL *, std::pair<unsigned int, unsigned int> > dl_map_;
    std::list<unsigned int> dl_freelist_;
    bool dl_lookup(GeomDL *obj, unsigned int &state, unsigned int &dl);
    bool dl_update(GeomDL *obj, unsigned int state);
    bool dl_addnew(GeomDL *obj, unsigned int state, unsigned int &dl);
    bool dl_remove(GeomDL *obj);

    bool mouse_action;
};

} // End namespace SCIRun


#endif /* SCI_Geom_GeomOpenGL_h */


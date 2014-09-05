  
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
#include <GL/glu.h>
#include <GL/glx.h>

#include <sci_config.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace PSECommon {
  namespace Modules {
    class Roe;
  }
}

namespace SCICore {
namespace GeomSpace {

using PSECommon::Modules::Roe;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

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
#ifndef _WIN32
    Display *dpy;
#endif
    int debug;
    void reset();
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.6.2.2  2000/10/26 17:18:37  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.7  2000/07/28 21:16:44  yarden
// add dli (display list)  flag to draw info
//
// Revision 1.6  1999/10/16 20:51:01  jmk
// forgive me if I break something -- this fixes picking and sets up sci
// bench - go to /home/sci/u2/VR/PSE for the latest sci bench technology
// gota getup to get down.
//
// Revision 1.5  1999/09/23 01:10:48  moulding
// added #include <stddef.h> and <stdlib.h> for types wchar_t needed by the
// win32 version of glu.h
//
// Revision 1.4  1999/08/28 17:54:41  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/19 05:30:55  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.2  1999/08/17 06:39:10  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:41  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:05  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:58  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:18  dav
// Import sources
//
//

#endif /* SCI_Geom_GeomOpenGL_h */


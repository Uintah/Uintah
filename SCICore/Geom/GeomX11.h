
/*
 *  GeomX11.h: Draw states for X11 renderers
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GeomX11_h
#define SCI_Geom_GeomX11_h 1

#include <Geom/GeomObj.h>
#include <Geom/Lighting.h>
#include <Geom/View.h>
#include <Geometry/Point.h>
#include <X11/Xlib.h>
//#include <Containers/Array1.h>
//#include <Containers/Stack.h>

namespace SCICore {
namespace Geometry {
class Transform;
}
}

namespace SCICore {
namespace GeomSpace {

  //void Pio();

using SCICore::Geometry::Transform;

class Light;
class Material;

struct DrawInfoX11 {
    DrawInfoX11();
    Material* current_matl;
    int current_lit;
    unsigned long current_pixel;

    unsigned long get_color(const Color&);
    void set_color(const Color&);

    int red_max;
    int green_max;
    int blue_max;
    int red_mult;
    int green_mult;
    unsigned long* colors;

    View view;
    Lighting lighting;

    Display* dpy;
    Window win;
    GC gc;
    Transform* transform;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:48  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:55  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:56:10  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:07  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_GeomX11_h */


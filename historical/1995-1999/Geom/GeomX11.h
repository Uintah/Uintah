
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

#include <Classlib/Array1.h>
#include <Classlib/Stack.h>
#include <Geom/Geom.h>
#include <Geom/Lighting.h>
#include <Geom/View.h>
#include <Geometry/Point.h>
#include <X11/Xlib.h>
class Light;
class Material;
class Transform;

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

#endif /* SCI_Geom_GeomX11_h */

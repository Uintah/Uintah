
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

#include <X11/Xlib.h>
#include <Classlib/Stack.h>
class Material;
class Transform;

struct DrawInfoX11 {
    DrawInfoX11();
    Material* current_matl;
    Stack<Material*> stack;
    void set_matl(Material*);
    void push_matl(Material*);
    void pop_matl();

    Display* dpy;
    Window win;
    GC gc;
    Transform* transform;
};

#endif /* SCI_Geom_GeomX11_h */


/*
 *  GeomX11.cc: Rendering for X windows
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/GeomX11.h>
#include <Geom/Geom.h>
#include <Geom/Line.h>
#include <Geom/Tri.h>
#include <Geometry/Transform.h>
#include <iostream.h>

DrawInfoX11::DrawInfoX11()
: current_matl(0)
{
}

void DrawInfoX11::set_matl(Material* matl)
{
//	NOT_FINISHED("Color calculation");
    XSetForeground(dpy, gc, WhitePixelOfScreen(DefaultScreenOfDisplay(dpy)));
}

void DrawInfoX11::push_matl(Material* matl)
{
    stack.push(matl);
    if(current_matl != matl){
	current_matl=matl;
	set_matl(matl);
    }
}

void DrawInfoX11::pop_matl()
{
    stack.pop();
    if(stack.size() > 0){
	Material* top=stack.top();
	if(current_matl != top){
	    current_matl=top;
	    set_matl(top);
	}
    } else {
	current_matl=0;
    }
}

void GeomObj::draw(DrawInfoX11* di)
{
    if(matl.get_rep())
	di->push_matl(matl.get_rep());
    objdraw(di);
    if(matl.get_rep())
	di->pop_matl();

}

double GeomObj::depth(DrawInfoX11*)
{
    cerr << "Error: depth called on an object which isn't a primitive!\n";
    return 0;
}

void GeomObj::objdraw(DrawInfoX11*)
{
    cerr << "Error: objdraw_X11 called on an object which isn't a primitive!\n";
}

double GeomTri::depth(DrawInfoX11* di)
{
    double d1=di->transform->project(p1).z();
    double d2=di->transform->project(p2).z();
    double d3=di->transform->project(p3).z();
    return Min(d1, d2, d3);
}

double GeomLine::depth(DrawInfoX11* di)
{
    double d1=di->transform->project(p1).z();
    double d2=di->transform->project(p2).z();
    return Min(d1, d2);
}

void GeomLine::objdraw(DrawInfoX11* di)
{
    Point t1(di->transform->project(p1));
    Point t2(di->transform->project(p2));
    int x1=Round(t1.x());
    int x2=Round(t2.x());
    int y1=Round(t1.y());
    int y2=Round(t2.y());
    XDrawLine(di->dpy, di->win, di->gc, x1, y1, x2, y2);
}

void GeomTri::objdraw(DrawInfoX11* di)
{
    Point t1(di->transform->project(p1));
    Point t2(di->transform->project(p2));
    Point t3(di->transform->project(p3));
    XPoint p[3];
    p[0].x=Round(t1.x());
    p[0].y=Round(t1.y());
    p[1].x=Round(t2.x());
    p[1].y=Round(t2.y());
    p[2].x=Round(t3.x());
    p[2].y=Round(t3.y());
    XFillPolygon(di->dpy, di->win, di->gc, p, 3, Convex,
		 CoordModeOrigin);
}


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
#include <Classlib/NotFinished.h>
#include <Geom/Geom.h>
#include <Geom/Light.h>
#include <Geom/Line.h>
#include <Geom/Tri.h>
#include <Geom/VCTri.h>
#include <Geometry/Transform.h>
#include <iostream.h>

DrawInfoX11::DrawInfoX11()
: current_matl(0)
{
}

void DrawInfoX11::set_color(const Color& c)
{
    int ri=Clamp(int(c.r()*red_max), 0, red_max);
    int gi=Clamp(int(c.g()*green_max), 0, green_max);
    int bi=Clamp(int(c.b()*blue_max), 0, blue_max);
    int idx=ri*red_mult+gi*green_mult+bi;
    unsigned long pixel=colors[idx];
    XSetForeground(dpy, gc, pixel);
    current_pixel=pixel;
}

void GeomObj::draw(DrawInfoX11* di)
{
    Material* old_matl=0;
    int old_lit=0;
    unsigned long old_pixel=0;
    if(lit || matl.get_rep() || lit != di->current_lit){
	old_matl=di->current_matl;
	old_lit=di->current_lit;
	if(matl.get_rep())
	    di->current_matl=matl.get_rep();
	di->current_lit=lit;
	old_pixel=di->current_pixel;
	if(lit){
	    // Get the normal and a representative poinnnt...
	    Vector normal;
	    Point hitpos;
	    get_hit(normal, hitpos);
	    Vector view_direction(hitpos-di->view.eyep);

	    // Start with the ambient light
	    Color result(di->current_matl->ambient*di->lighting.amblight);

	    // Add the object's emission
	    result+=di->current_matl->emission;
	    Color difflight(0,0,0);
	    Color speclight(0,0,0);
	    int nlights=di->lighting.lights.size();
	    for(int i=0;i<nlights;i++){
		// Compute diffuse contribution
		Vector light_dir;
		Color light;
		di->lighting.lights[i]->compute_lighting(di->view, hitpos,
							 light, light_dir);

		double cos_theta=Dot(light_dir, normal);
		if(cos_theta < 0){
		    // Flip normal...
		    cos_theta=-cos_theta;
		    normal=-normal;
		}
		difflight+=light*cos_theta;
		// Compute specular contribution

		Vector H(light_dir+view_direction);
		H.normalize();
		double cos_alpha=Dot(H, normal);
		if(cos_alpha > 0.0 && di->current_matl->shininess > 0.0){
		    speclight+=light*Pow(cos_alpha, di->current_matl->shininess);
		}
	    }
	    result+=difflight*di->current_matl->diffuse;
	    result+=speclight*di->current_matl->specular;
	    di->set_color(result);
	} else {
	    // Just use the diffuse color to get the pixel...
	    di->set_color(di->current_matl->diffuse);
	}
    }
    objdraw(di);
    if(old_matl){
	di->current_matl=old_matl;
	di->current_lit=old_lit;
	if(!lit){
	    XSetForeground(di->dpy, di->gc, old_pixel);
	    di->current_pixel=old_pixel;
	}
    }
}

double GeomObj::depth(DrawInfoX11*)
{
    cerr << "Error: depth called on an object which isn't a primitive!\n";
    return 0;
}

void GeomObj::objdraw(DrawInfoX11*)
{
    cerr << "Error: objdraw called on an object which isn't a primitive!\n";
}
void GeomObj::get_hit(Vector&, Point&)
{
    cerr << "Error: get_hit called on an object which isn't a lit primitive!\n";
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

double GeomTri::depth(DrawInfoX11* di)
{
    double d1=di->transform->project(p1).z();
    double d2=di->transform->project(p2).z();
    double d3=di->transform->project(p3).z();
    return Min(d1, d2, d3);
}

void GeomTri::get_hit(Vector& normal, Point& point)
{
    point=AffineCombination(p1, 1./3., p2, 1./3., p3, 1./3.);
    normal=n;
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

double GeomVCTri::depth(DrawInfoX11* di)
{
    double d1=di->transform->project(p1).z();
    double d2=di->transform->project(p2).z();
    double d3=di->transform->project(p3).z();
    return Min(d1, d2, d3);
}

void GeomVCTri::get_hit(Vector& normal, Point& point)
{
    point=AffineCombination(p1, 1./3., p2, 1./3., p3, 1./3.);
    normal=n;
}

void GeomVCTri::objdraw(DrawInfoX11* di)
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

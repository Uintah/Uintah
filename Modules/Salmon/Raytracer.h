
/*
 *  Raytracer.h: A raytracer for Salmon
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_Modules_Salmon_Raytracer_h
#define sci_Modules_Salmon_Raytracer_h 1

#include <Modules/Salmon/Renderer.h>
#include <Geom/Color.h>
#include <Geom/Material.h>

class Hit;
class Ray;

class Raytracer : public Renderer {
    Renderer* rend;
    char* strbuf;
    Color bgcolor;
    int bg_firstonly;
    GeomObj* topobj;
    Salmon* salmon;
    Roe* roe;
    int max_level;
    double min_weight;
    View* current_view;
    MaterialHandle topmatl;

    Color trace_ray(const Ray& ray, int level, double weight,
		    double ior);
    Color shade(const Ray& ray, const Hit&, int level, double weight,
		double ior);
    void inside_out(int n, int a, int& b, int& r);
public:
    Raytracer();
    virtual ~Raytracer();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&);
    virtual void hide();
    virtual void put_scanline(int y, int width, Color* scanline, int repeat);
    double light_ray(const Point& from, const Point& to,
		     const Vector& direction, double dist);
};

#endif


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

namespace SCICore {
  namespace Geometry {
    class Point;
    class Vector;
    class Ray;
  }
  namespace GeomSpace {
    class GeomGroup;
    class Hit;
  }
}

namespace PSECommon {
namespace Modules {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Ray;
using SCICore::GeomSpace::Hit;
using SCICore::GeomSpace::GeomGroup;
using SCICore::GeomSpace::MaterialHandle;

class Raytracer : public Renderer {
    Renderer* rend;
    char* strbuf;
    Color bgcolor;
    int bg_firstonly;
    GeomGroup* topobj;
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

    void add_to_group(GeomObj*);
public:
    Raytracer();
    virtual ~Raytracer();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void old_redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&, int&);
    virtual void hide();
    virtual void put_scanline(int y, int width, Color* scanline, int repeat);
    double light_ray(const Point& from, const Point& to,
		     const Vector& direction, double dist);
};

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:52  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:09  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//

#endif

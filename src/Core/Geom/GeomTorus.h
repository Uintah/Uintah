
/*
 * GeomTorus.h: Torus objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Torus_h
#define SCI_Geom_Torus_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomTorus : public GeomObj {
public:
    Point cen;
    Vector axis;
    double rad1;
    double rad2;
    int nu;
    int nv;

    Vector zrotaxis;
    double zrotangle;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);

    GeomTorus(int nu=50, int nv=8);
    GeomTorus(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);
    GeomTorus(const GeomTorus&);
    virtual ~GeomTorus();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomTorusArc : public GeomTorus {
public:
    Vector zero;
    double start_angle;
    double arc_angle;
    Vector yaxis;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      const Vector& zero, double start_angle, double arc_angle,
	      int nu=50, int nv=8);
    GeomTorusArc(int nu=50, int nv=8);
    GeomTorusArc(const Point&, const Vector&, double, double, 
		 const Vector& zero, double start_angle, double arc_angle,
		 int nu=50, int nv=8);
    GeomTorusArc(const GeomTorusArc&);
    virtual ~GeomTorusArc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:46  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:27  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:15  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:46  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:09  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:04  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_Torus_h */

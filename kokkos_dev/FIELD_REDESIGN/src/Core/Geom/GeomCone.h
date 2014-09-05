
/*
 *  Cone.h: Cone object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Cone_h
#define SCI_Geom_Cone_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomCone : public GeomObj {
protected:
    Vector v1;
    Vector v2;
    double tilt;
    double height;
    Vector zrotaxis;
    double zrotangle;
public:
    Point bottom;
    Point top;
    Vector axis;
    double bot_rad;
    double top_rad;
    int nu;
    int nv;

    void adjust();
    void move(const Point&, const Point&, double, double, int nu=20, int nv=1);

    GeomCone(int nu=20, int nv=1);
    GeomCone(const Point&, const Point&, double, double, int nu=20, int nv=1);
    GeomCone(const GeomCone&);
    virtual ~GeomCone();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomCappedCone : public GeomCone {
    int nvdisc1;
    int nvdisc2;
public:
    GeomCappedCone(int nu=20, int nv=1, int nvdisc1=1, int nvdisc2=1);
    GeomCappedCone(const Point&, const Point&, double, double, 
		   int nu=20, int nv=1, int nvdisc1=1, int nvdisc2=1);
    GeomCappedCone(const GeomCappedCone&);
    virtual ~GeomCappedCone();

    virtual GeomObj* clone();
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
// Revision 1.5  1999/11/19 06:43:25  dmw
// fewer triangles in default size for cones and cylinders
//
// Revision 1.4  1999/10/07 02:07:41  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:19  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:06  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:38  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:03  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:55  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Cone_h */


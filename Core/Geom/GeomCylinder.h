
/*
 *  Cylinder.h: Cylinder Object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Cylinder_h
#define SCI_Geom_Cylinder_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::BBox;

class SCICORESHARE GeomCylinder : public GeomObj {
protected:
    Vector v1;
    Vector v2;

    double height;
    Vector zrotaxis;
    double zrotangle;
public:
    Point bottom;
    Point top;
    Vector axis;
    double rad;
    int nu;
    int nv;
    void adjust();
    void move(const Point&, const Point&, double, int nu=20, int nv=10);

    GeomCylinder(int nu=20, int nv=10);
    GeomCylinder(const Point&, const Point&, double, int nu=20, int nv=10);
    GeomCylinder(const GeomCylinder&);
    virtual ~GeomCylinder();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

class SCICORESHARE GeomCappedCylinder : public GeomCylinder {
    int nvdisc;
public:
    GeomCappedCylinder(int nu=20, int nv=10, int nvdisc=4);
    GeomCappedCylinder(const Point&, const Point&, double, int nu=20, int nv=10, int nvdisc=4);
    GeomCappedCylinder(const GeomCappedCylinder&);
    virtual ~GeomCappedCylinder();

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
// Revision 1.4  1999/10/07 02:07:41  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:20  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:39  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:04  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:56  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Cylinder_h */

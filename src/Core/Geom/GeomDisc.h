
/*
 *  Disc.h:  Disc object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Disc_h
#define SCI_Geom_Disc_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomDisc : public GeomObj {
    Vector v1;
    Vector v2;
    Vector zrotaxis;
    double zrotangle;
public:
    Point cen;
    Vector n;
    double rad;
    int nu;
    int nv;

    void adjust();
    void move(const Point&, const Vector&, double, int nu=20, int nv=2);

    GeomDisc(int nu=20, int nv=2);
    GeomDisc(const Point&, const Vector&, double, int nu=20, int nv=2);
    GeomDisc(const GeomDisc&);
    virtual ~GeomDisc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
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
// Revision 1.1  1999/05/05 21:04:57  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Disc_h */


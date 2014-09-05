
/*
 *  Tetra.h:  A tetrahedra object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tetra_h
#define SCI_Geom_Tetra_h 1

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomTetra : public GeomObj {
public:
    Point p1;
    Point p2;
    Point p3;
    Point p4;

    GeomTetra(const Point&, const Point&, const Point&, const Point&);
    GeomTetra(const GeomTetra&);
    virtual ~GeomTetra();

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
// Revision 1.4  1999/10/07 02:07:45  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:25  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:44  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:08  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:02  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//

#endif /* SCI_Geom_Tetra_h */

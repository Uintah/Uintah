
/*
 *  Tri.h: Triangles...
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Tri_h
#define SCI_Geom_Tri_h 1

#include <SCICore/Geom/GeomVertexPrim.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomTri : public GeomVertexPrim {
    Vector n;
public:
    GeomTri(const Point&, const Point&, const Point&);
    GeomTri(const Point&, const Point&, const Point&,
	    const MaterialHandle&,
	    const MaterialHandle&,
	    const MaterialHandle&);
    GeomTri(const GeomTri&);
    virtual ~GeomTri();

    virtual GeomObj* clone();

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
// Revision 1.3  1999/08/17 23:50:28  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:16  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:46  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:09  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:05  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:18  dav
// Import sources
//
//

#endif /* SCI_Geom_Tri_h */

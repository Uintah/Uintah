
/*
 *  GeomContainer.h: Base class for container objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Container_h
#define SCI_Geom_Container_h 1

#include <SCICore/Geom/GeomObj.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomContainer : public GeomObj {
protected:
    GeomObj* child;
public:
    GeomContainer(GeomObj*);
    GeomContainer(const GeomContainer&);
    virtual ~GeomContainer();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    // For all Painter's algorithm based renderers
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);

    // For Raytracing
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material* matl,
			   Hit& hit);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};    

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:38  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:04  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:56  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif

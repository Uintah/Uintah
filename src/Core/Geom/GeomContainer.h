
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

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_triangles( Array1<float> &);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};    

} // End namespace GeomSpace
} // End namespace SCICore


#endif

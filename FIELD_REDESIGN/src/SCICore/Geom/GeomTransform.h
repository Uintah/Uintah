/*
 *  Transform.h:  Transform Properities for Geometry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Geom_Transform_h
#define SCI_Geom_Transform_h 1

#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geom/GeomContainer.h>
#include <SCICore/Geometry/Transform.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Transform;

class SCICORESHARE GeomTransform : public GeomContainer {
    Transform trans;
public:
    GeomTransform(GeomObj*);
    GeomTransform(GeomObj*, const Transform);
    GeomTransform(const GeomTransform&);
    void setTransform(const Transform);
    Transform getTransform();
    virtual ~GeomTransform();
    virtual GeomObj* clone();

    virtual void get_bounds(BBox&);

    void scale(const Vector&);
    void translate(const Vector&);
    void rotate(double, const Vector&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCICore
} // End namespace GeomSpace

#endif


/*
 *  Group.h:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Group_h
#define SCI_Geom_Group_h 1

#include <SCICore/share/share.h>

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

class SCICORESHARE GeomGroup : public GeomObj {
    Array1<GeomObj*> objs;
    int del_children;

public:
    GeomGroup(int del_children=1);
    GeomGroup(const GeomGroup&);
    virtual ~GeomGroup();
    virtual GeomObj* clone();

    void add(GeomObj*);
    void remove(GeomObj*);
    void remove_all();
    int size();

    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_triangles( Array1<float> &);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

#endif /* SCI_Geom_Group_h */

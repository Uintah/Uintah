
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
#include <SCICore/Geometry/BSphere.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

struct ITree;

class SCICORESHARE GeomGroup : public GeomObj {
    Array1<GeomObj*> objs;
    BSphere bsphere;
    int del_children;

    ITree* treetop;
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
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:05  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:57  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Group_h */

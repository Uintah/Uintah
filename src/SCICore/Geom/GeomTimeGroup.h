
/*
 *  GeomTimeGroup.h:  ?
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Time__Group_h
#define SCI_Geom_Time_Group_h 1

#include <Geom/GeomObj.h>
#include <Geometry/BSphere.h>
#include <Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

class GeomTimeGroup : public GeomObj {
    Array1<GeomObj*> objs;
    Array1<double>   start_times;
    int del_children;

    BBox bbox; // bbox for entire seen - set once!
public:
    GeomTimeGroup(int del_children=1);
    GeomTimeGroup(const GeomTimeGroup&);
    virtual ~GeomTimeGroup();
    virtual GeomObj* clone();

    void add(GeomObj*,double); // with time...
    void remove(GeomObj*);
    void remove_all();
    int size();

    void  setbbox(BBox&); // sets bounding box - so isn't computed!

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
// Revision 1.1  1999/07/27 16:56:45  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:08  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:03  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif /* SCI_Geom_Group_h */

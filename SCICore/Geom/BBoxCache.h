
/*
 *  BBoxCache.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_Geom_BBoxCache_h 
#define SCI_Geom_BBoxCache_h 1

#include <Geom/GeomObj.h>
#include <Containers/String.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

namespace SCICore {
namespace GeomSpace {

class GeomBBoxCache: public GeomObj {
    GeomObj* child;

    int bbox_cached;
    int bsphere_cached;
    BBox bbox;
    BSphere bsphere;
public:
    GeomBBoxCache(GeomObj*);
    GeomBBoxCache(GeomObj*, BBox &);

    virtual ~GeomBBoxCache();

    virtual GeomObj* clone();
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
// Revision 1.1  1999/07/27 16:56:36  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:02  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:54  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:19  dav
// Import sources
//
//

#endif


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

#include <Core/Geom/GeomObj.h>
#include <Core/Containers/String.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomBBoxCache: public GeomObj {
    GeomObj* child;

    int bbox_cached;
    BBox bbox;
public:
    GeomBBoxCache(GeomObj*);
    GeomBBoxCache(GeomObj*, const BBox &);

    virtual ~GeomBBoxCache();

    virtual GeomObj* clone();
    virtual void reset_bbox();
    virtual void get_bounds(BBox&);
    
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_triangles(Array1<float> &);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;	
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun

#endif


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

#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/BBox.h>

namespace SCICore {
namespace GeomSpace {

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4.2.2  2000/10/26 17:18:35  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.6  2000/07/06 23:18:55  yarden
// fix a bug if the bbox is not valid
//
// Revision 1.5  2000/06/06 16:01:42  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.4  1999/10/07 02:07:40  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/17 23:50:18  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:04  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

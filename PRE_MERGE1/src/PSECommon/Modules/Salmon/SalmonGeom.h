#ifndef SCI_Salmon_Geom_h
#define SCI_Salmon_Geom_h 1

/*
 *  SalmonGeom.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Geom/IndexedGroup.h>
#include <Geom/GeomObj.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/Material.h>
#include <Geom/GeomSave.h>
#include <Geometry/Ray.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>
#include <Containers/String.h>
#include <Containers/Array1.h>
#include <Persistent/Persistent.h>
#include <Dataflow/Module.h>

namespace SCICore {
  namespace Multitask {
    class CrowdMonitor;
  }
}

namespace PSECommon {
  namespace CommonDatatypes {
    class GeometryComm;
  }
}

namespace PSECommon {
namespace Modules {

using PSECommon::Dataflow::Module;
using PSECommon::CommonDatatypes::GeometryComm;

using SCICore::GeomSpace::GeomIndexedGroup;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::DrawInfoOpenGL;
using SCICore::GeomSpace::Material;
using SCICore::GeomSpace::GeomSave;
using SCICore::GeomSpace::Hit;
using SCICore::Geometry::Ray;
using SCICore::Geometry::BSphere;
using SCICore::Geometry::BBox;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Multitask::CrowdMonitor;

/* this is basicaly a indexed group that also has some simple message
 * stuff
 */	

class GeomSalmonPort: public GeomIndexedGroup {
    GeometryComm* msg_head;
    GeometryComm* msg_tail;

    int portno;
    
public:
    friend class Salmon;
    GeomSalmonPort(int);
    virtual ~GeomSalmonPort();

    GeometryComm* getHead(void) { return msg_head; }
};

/*
 * for items in a scene - has name (for roes to lookup)
 * a lock and a geomobj (to call)
 */

class GeomSalmonItem: public GeomObj {
    GeomObj *child;
    clString name;
    CrowdMonitor* lock;

public:
    friend class Roe;
    GeomSalmonItem();
    GeomSalmonItem(GeomObj*,const clString&, CrowdMonitor* lock);
    virtual ~GeomSalmonItem();

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
    bool saveobj(ostream& out, const clString& format,
		 GeomSave* saveinfo);
    
    clString& getString(void) { return name;}
};

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:11  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//

#endif

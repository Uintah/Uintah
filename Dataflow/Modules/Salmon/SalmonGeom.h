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

#include <SCICore/Geom/IndexedGroup.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomSave.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Persistent/Persistent.h>
#include <PSECore/Dataflow/Module.h>

namespace SCICore {
  namespace Thread {
    class CrowdMonitor;
  }
}

namespace PSECore {
  namespace Datatypes {
    class GeometryComm;
  }
}

namespace PSECommon {
namespace Modules {

using PSECore::Dataflow::Module;
using PSECore::Datatypes::GeometryComm;

using SCICore::GeomSpace::GeomIndexedGroup;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::DrawInfoOpenGL;
using SCICore::GeomSpace::Material;
using SCICore::GeomSpace::GeomSave;
using SCICore::GeomSpace::Hit;
using SCICore::Geometry::BBox;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

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
    SCICore::Thread::CrowdMonitor* lock;

public:
    friend class Roe;
    GeomSalmonItem();
    GeomSalmonItem(GeomObj*,const clString&, SCICore::Thread::CrowdMonitor* lock);
    virtual ~GeomSalmonItem();

    virtual GeomObj* clone();
    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
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
// Revision 1.5  1999/08/29 00:46:43  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/25 03:47:58  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/17 23:50:16  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:37:40  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

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
  virtual void get_triangles( Array1<float> &);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  bool saveobj(std::ostream& out, const clString& format,
	       GeomSave* saveinfo);
    
  clString& getString(void) { return name;}
};

} // End namespace Modules
} // End namespace PSECommon


#endif

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

#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSave.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Persistent.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {
  class CrowdMonitor;
}

namespace SCIRun {
  class GeometryComm;
}

namespace SCIRun {



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

} // End namespace SCIRun


#endif

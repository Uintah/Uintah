/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef SCI_Render_ViewGeom_h
#define SCI_Render_ViewGeom_h 1

/*
 *  ViewGeom.h: ?
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
#include <Core/Persistent/Persistent.h>
#include <Dataflow/Network/Module.h>


namespace SCIRun {

class CrowdMonitor;
class GeometryComm;

/* this is basicaly a indexed group that also has some simple message
 * stuff
 */	

class GeomViewerPort: public GeomIndexedGroup {
  GeometryComm* msg_head;
  GeometryComm* msg_tail;

  int portno;
    
public:
  friend class Viewer;
  GeomViewerPort(int);
  virtual ~GeomViewerPort();

  GeometryComm* getHead(void) { return msg_head; }
};

/*
 * for items in a scene - has name (for roes to lookup)
 * a lock and a geomobj (to call)
 */

class GeomViewerItem: public GeomObj {
  GeomObj *child;
  string name;
  CrowdMonitor* lock;

public:
  friend class ViewWindow;
  GeomViewerItem();
  GeomViewerItem(GeomObj*,const string&, CrowdMonitor* lock);
  virtual ~GeomViewerItem();

  virtual GeomObj* clone();
  virtual void reset_bbox();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void get_triangles( Array1<float> &);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  bool saveobj(std::ostream& out, const string& format,
	       GeomSave* saveinfo);
    
  string& getString(void) { return name;}
};

} // End namespace SCIRun


#endif

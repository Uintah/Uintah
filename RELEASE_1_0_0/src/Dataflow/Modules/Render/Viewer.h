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


/*
 *  Viewer.h: The Geometry Viewer!
 *
 *  Written by:
 *   Steven G. Parker & Dave Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Viewer_h
#define SCI_project_module_Viewer_h

#include <Dataflow/Network/Module.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Lighting.h>
#include <Core/Geom/IndexedGroup.h>
#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/CrowdMonitor.h>

#include <map.h>

namespace SCIRun {



class Renderer;
class ViewWindow;

#if 0
struct SceneItem {
  GeomObj* obj;
  clString name;
  CrowdMonitor* lock;

  SceneItem(GeomObj*, const clString&, CrowdMonitor* lock);
  ~SceneItem();
};

struct PortInfo {
  GeometryComm* msg_head;
  GeometryComm* msg_tail;
  int portno;

  typedef map<int, SceneItem*> MapIntSceneItem;
  MapIntSceneItem* objs;
};
#endif

class Viewer : public Module {
    
public:
  typedef map<clString, void*>	MapClStringVoid;
#if 0    
  typedef map<int, PortInfo*>		MapIntPortInfo;
#endif

private:
  Array1<ViewWindow*> viewwindow;
  int busy_bit;
  Array1<ViewWindow*> topViewWindow;
  virtual void do_execute();

  int max_portno;
  //virtual void connection(Module::ConnectionMode, int, int);

  MapClStringVoid specific;
    
public:
  MaterialHandle default_matl;
  friend class ViewWindow;
  Viewer(const clString& id);
  Viewer(const clString& id, const clString& moduleName);
  virtual ~Viewer();
  virtual void execute();
  void initPort(Mailbox<GeomReply>*);
  void append_port_msg(GeometryComm*);
  void addObj(GeomViewerPort* port, GeomID serial, GeomObj *obj,
	      const clString&, CrowdMonitor* lock);
  void delObj(GeomViewerPort* port, GeomID serial, int del);
  void delAll(GeomViewerPort* port);
  void flushPort(int portid);
  void flushViews();
  void addTopViewWindow(ViewWindow *r);
  void delTopViewWindow(ViewWindow *r);

  void delete_viewwindow(ViewWindow* r);

  void tcl_command(TCLArgs&, void*);

  virtual void emit_vars(std::ostream& out, 
			 clString& midx); // Override from class TCL

				// The scene...
  GeomIndexedGroup ports;	// this contains all of the ports...

#if 0    
  MapIntPortInfo portHash;
#endif

				// Lighting
  Lighting lighting;

  int process_event(int block);

  int lookup_specific(const clString& key, void*&);
  void insert_specific(const clString& key, void* data);

  CrowdMonitor geomlock;
};

class ViewerMessage : public MessageBase {
public:
  clString rid;
  clString filename;
  clString format;
  double tbeg, tend;
  int nframes;
  double framerate;
  ViewerMessage(const clString& rid);
  ViewerMessage(const clString& rid, double tbeg, double tend,
		int nframes, double framerate);
  ViewerMessage(MessageTypes::MessageType,
		const clString& rid, const clString& filename);
  ViewerMessage(MessageTypes::MessageType,
		const clString& rid, const clString& filename,
		const clString& format);
  virtual ~ViewerMessage();
};

} // End namespace SCIRun

#endif

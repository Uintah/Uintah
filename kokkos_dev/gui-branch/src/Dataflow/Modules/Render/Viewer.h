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
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Lighting.h>
#include <Core/Geom/IndexedGroup.h>
#include <Dataflow/Modules/Render/ViewGeom.h>
#include <Core/GuiInterface/TCLArgs.h>
#include <Core/Thread/CrowdMonitor.h>

#include <map>
#include <vector>

namespace SCIRun {

using std::vector;

class Renderer;
class ViewWindow;


class Viewer : public Module {
    
public:
  typedef map<string, void*> MapStringVoid;

private:
  vector<ViewWindow*> viewwindow;
  int busy_bit;
  vector<ViewWindow*> topViewWindow;
  virtual void do_execute();

  int max_portno;

  MapStringVoid specific;
    
public:
  MaterialHandle default_matl;
  friend class ViewWindow;
  Viewer(const string& id);
  Viewer(const string& id, const string& moduleName);
  virtual ~Viewer();
  virtual void execute();
  void initPort(Mailbox<GeomReply>*);
  void append_port_msg(GeometryComm*);
  void addObj(GeomViewerPort* port, GeomID serial, GeomObj *obj,
	      const string&, CrowdMonitor* lock);
  void delObj(GeomViewerPort* port, GeomID serial, int del);
  void delAll(GeomViewerPort* port);
  void flushPort(int portid);
  void flushViews();
  void addTopViewWindow(ViewWindow *r);
  void delTopViewWindow(ViewWindow *r);

  void delete_viewwindow(ViewWindow* r);

  void tcl_command(TCLArgs&, void*);

  virtual void emit_vars(std::ostream& out, 
			 string& midx); // Override from class TCL

				// The scene...
  GeomIndexedGroup ports;	// this contains all of the ports...
  Lighting lighting;            // Lighting

  int process_event(int block);

  int lookup_specific(const string& key, void*&);
  void insert_specific(const string& key, void* data);

  CrowdMonitor geomlock;
};

class ViewerMessage : public MessageBase {
public:
  string rid;
  string filename;
  string format;
  double tbeg, tend;
  int nframes;
  double framerate;
  ViewerMessage(const string& rid);
  ViewerMessage(const string& rid, double tbeg, double tend,
		int nframes, double framerate);
  ViewerMessage(MessageTypes::MessageType,
		const string& rid, const string& filename);
  ViewerMessage(MessageTypes::MessageType,
		const string& rid, const string& filename,
		const string& format);
  virtual ~ViewerMessage();
};

} // End namespace SCIRun

#endif

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
#include <Core/Thread/CrowdMonitor.h>

#include <map>
#include <vector>

namespace SCIRun {

using std::vector;
using std::map;

class ViewWindow;

class Viewer : public Module {
public:

  Viewer(GuiContext*);
  virtual ~Viewer();
  virtual void do_execute();
  virtual void execute();

  void delete_viewwindow(ViewWindow* r);

  MaterialHandle      default_material_;
  Lighting            lighting_;
  map<int, map<LightID, int> > pli_;  // port->light->index

  CrowdMonitor        geomlock_;
  GeomIndexedGroup    ports_;

  // CollabVis code begin
#ifdef HAVE_COLLAB_VIS
  Mailbox<ViewWindow*> newViewWindowMailbox;
#endif 
  // CollabVis code end
  
private:
  
  void initPort(Mailbox<GeomReply>*);
  void detachPort(int portno);
  int  real_portno(int portid);
  void delete_patch_portnos(int portid);
  void append_port_msg(GeometryComm*);
  void addObj(GeomViewerPort* port, GeomID serial, GeomHandle obj,
	      const string&, CrowdMonitor* lock);
  void delObj(GeomViewerPort* port, GeomID serial);
  void delAll(GeomViewerPort* port);
  void flushPort(int portid);
  void flushViews();
  void addTopViewWindow(ViewWindow *r);
  void delTopViewWindow(ViewWindow *r);

  void tcl_command(GuiArgs&, void*);

  virtual void emit_vars(std::ostream& out,
			 const string& midx); // Override from class Module
  int process_event();

  vector<ViewWindow*> view_window_;
  vector<ViewWindow*> top_view_window_;
  int                 max_portno_;
  vector<int>         portno_map_;
  vector<bool>        syncronized_map_;
};



class ViewerMessage : public MessageBase {
public:
  string rid;
  string filename;
  string format;
  int resx;
  int resy;
  double tbeg, tend;
  int nframes;
  double framerate;
  Vector lightDir;
  Color lightColor;
  int lightNo;
  bool on;

  ViewerMessage(const string& rid);
  ViewerMessage(const string& rid, double tbeg, double tend,
		int nframes, double framerate);
  ViewerMessage(MessageTypes::MessageType,
		const string& rid, int lightNo, bool on, 
		const Vector& dir, const Color& color);
  ViewerMessage(MessageTypes::MessageType,
		const string& rid, const string& filename);
  ViewerMessage(MessageTypes::MessageType,
		const string& rid, const string& filename,
		const string& format, const string &resx_string,
		const string& resy_string);
  virtual ~ViewerMessage();
};

} // End namespace SCIRun

#endif

/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#ifndef SCIRun_src_Dataflow_Modules_Render_Viewer_h
#define SCIRun_src_Dataflow_Modules_Render_Viewer_h

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
  virtual void			do_execute();
  virtual void			execute();

  MaterialHandle		default_material_;
  Lighting			lighting_;
  CrowdMonitor			geomlock_;
  GeomIndexedGroup		ports_;

private:
  void				tcl_command(GuiArgs&, void*);  
  void				delete_viewwindow(const string &id);
  void				initPort(Mailbox<GeomReply>*);
  void				detachPort(int portno);
  int				real_portno(int portid);
  void				delete_patch_portnos(int portid);
  void				append_port_msg(GeometryComm*);
  void				addObj(GeomViewerPort* port, 
				       GeomID serial, GeomHandle obj,
				       const string&, CrowdMonitor* lock);
  void				delObj(GeomViewerPort* port, GeomID serial);
  void				delAll(GeomViewerPort* port);
  void				flushPort(int portid);
  void				finishPort(int portid);
  void				flushViews();
  int				process_event();

  virtual void set_context(Scheduler* sched, Network* network);

  map<int, map<LightID, int> >	pli_;  // port->light->index
  Mutex				view_window_lock_;
  vector<ViewWindow*>		view_window_;
  int				max_portno_;
  bool				stop_rendering_;
  vector<int>			portno_map_;
  vector<int>			synchronized_map_;
  list<unsigned int>            synchronized_serials_;
  int                           synchronized_debt_;


  static bool regression_callback(void *ths);
};



class ViewerMessage : public MessageBase {
public:
  string			rid;
  string			filename;
  string			format;
  int				resx;
  int				resy;
  double			tbeg;
  double			tend;
  int				nframes;
  double			framerate;
  Vector			lightDir;
  Color				lightColor;
  int				lightNo;
  bool				on;

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

#endif // of #ifndef SCIRun_src_Dataflow_Modules_Render_Viewer_h

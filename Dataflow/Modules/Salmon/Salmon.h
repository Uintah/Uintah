
/*
 *  Salmon.h: The Geometry Viewer!
 *
 *  Written by:
 *   Steven G. Parker & Dave Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Salmon_h
#define SCI_project_module_Salmon_h

#include <Dataflow/Network/Module.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Lighting.h>
#include <Core/Geom/IndexedGroup.h>
#include <Dataflow/Modules/Salmon/SalmonGeom.h>
#include <Core/TclInterface/TCL.h>
#include <Core/Thread/CrowdMonitor.h>

#include <map.h>

namespace SCIRun {



class Renderer;
class Roe;

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

class Salmon : public Module {
    
public:
  typedef map<clString, void*>	MapClStringVoid;
#if 0    
  typedef map<int, PortInfo*>		MapIntPortInfo;
#endif

private:
  Array1<Roe*> roe;
  int busy_bit;
  Array1<Roe*> topRoe;
  virtual void do_execute();

  int max_portno;
  virtual void connection(Module::ConnectionMode, int, int);

  MapClStringVoid specific;
    
public:
  MaterialHandle default_matl;
  friend class Roe;
  Salmon(const clString& id);
  Salmon(const clString& id, const clString& moduleName);
  virtual ~Salmon();
  virtual void execute();
  void initPort(Mailbox<GeomReply>*);
  void append_port_msg(GeometryComm*);
  void addObj(GeomSalmonPort* port, GeomID serial, GeomObj *obj,
	      const clString&, CrowdMonitor* lock);
  void delObj(GeomSalmonPort* port, GeomID serial, int del);
  void delAll(GeomSalmonPort* port);
  void flushPort(int portid);
  void flushViews();
  void addTopRoe(Roe *r);
  void delTopRoe(Roe *r);

  void delete_roe(Roe* r);

  void tcl_command(TCLArgs&, void*);

  virtual void emit_vars(std::ostream& out); // Override from class TCL

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

class SalmonMessage : public MessageBase {
public:
  clString rid;
  clString filename;
  clString format;
  double tbeg, tend;
  int nframes;
  double framerate;
  SalmonMessage(const clString& rid);
  SalmonMessage(const clString& rid, double tbeg, double tend,
		int nframes, double framerate);
  SalmonMessage(MessageTypes::MessageType,
		const clString& rid, const clString& filename);
  SalmonMessage(MessageTypes::MessageType,
		const clString& rid, const clString& filename,
		const clString& format);
  virtual ~SalmonMessage();
};

} // End namespace SCIRun

#endif

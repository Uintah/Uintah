
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

#include <Dataflow/Module.h>
#include <Comm/MessageBase.h>
#include <Containers/Array1.h>
#include <Containers/HashTable.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/GeometryComm.h>
#include <Geom/GeomObj.h>
#include <Geom/Material.h>
#include <Geom/Lighting.h>
#include <Multitask/ITC.h>
#include <Geom/IndexedGroup.h>
#include <Modules/Salmon/SalmonGeom.h>
#include <TclInterface/TCL.h>

namespace PSECommon {
namespace Modules {

using PSECommon::CommonDatatypes::GeomID;
using PSECommon::Comm::MessageBase;
using PSECommon::Comm::MessageTypes;
using PSECommon::CommonDatatypes::GeomReply;

using SCICore::Containers::HashTable;
using SCICore::Multitask::Mailbox;
using SCICore::Multitask::CrowdMonitor;
using SCICore::GeomSpace::MaterialHandle;
using SCICore::TclInterface::TCLArgs;
using SCICore::GeomSpace::Lighting;

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
    HashTable<int, SceneItem*>* objs;
};
#endif

class Salmon : public Module {
    Array1<Roe*> roe;
    int busy_bit;
    Array1<Roe*> topRoe;
    virtual void do_execute();

    int max_portno;
    virtual void connection(Module::ConnectionMode, int, int);

    HashTable<clString, void*> specific;
public:
    MaterialHandle default_matl;
    friend class Roe;
    Salmon(const clString& id);
    Salmon(const Salmon&, int deep);
    virtual ~Salmon();
    virtual Module* clone(int deep);
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

    virtual void emit_vars(ostream& out); // Override from class TCL

    // The scene...
    GeomIndexedGroup ports; // this contains all of the ports...
//    HashTable<int, PortInfo*> portHash;

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:10  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//

#endif

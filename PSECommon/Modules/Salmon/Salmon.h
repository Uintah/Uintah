
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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Comm/MessageBase.h>
#include <SCICore/Containers/Array1.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/GeometryComm.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Lighting.h>
#include <SCICore/Geom/IndexedGroup.h>
#include <PSECommon/Modules/Salmon/SalmonGeom.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/Thread/CrowdMonitor.h>

namespace PSECommon {
namespace Modules {

using PSECore::Datatypes::GeomID;
using PSECore::Comm::MessageBase;
using PSECore::Comm::MessageTypes;
using PSECore::Datatypes::GeomReply;

using SCICore::GeomSpace::MaterialHandle;
using SCICore::TclInterface::TCLArgs;
using SCICore::GeomSpace::Lighting;
using SCICore::Thread::CrowdMonitor;

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

    typedef map< int, SceneItem*, less<int> > MapIntSceneItem;
    MapIntSceneItem* objs;
};
#endif

class Salmon : public Module {
    
public:
    typedef map< clString, void*, less<clString> >	MapClStringVoid;
#if 0    
    typedef map< int, PortInfo*, less<int> >		MapIntPortInfo;
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
    virtual ~Salmon();
    virtual void execute();
    void initPort(SCICore::Thread::Mailbox<GeomReply>*);
    void append_port_msg(GeometryComm*);
    void addObj(GeomSalmonPort* port, GeomID serial, GeomObj *obj,
		const clString&, SCICore::Thread::CrowdMonitor* lock);
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

    SCICore::Thread::CrowdMonitor geomlock;
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
// Revision 1.7  2000/03/11 00:39:53  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.6  1999/10/07 02:06:57  sparker
// use standard iostreams and complex type
//
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
// Revision 1.3  1999/08/18 20:19:53  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:39  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

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
#include <Classlib/Array1.h>
#include <Classlib/HashTable.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geom/Lighting.h>
#include <Multitask/ITC.h>

class GeometryComm;
class GeomReply;
class Renderer;
class Roe;

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

class Salmon : public Module {
    Array1<Roe*> roe;
    int busy_bit;
    Array1<Roe*> topRoe;
    virtual void do_execute();
    virtual int should_execute();

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
    void addObj(PortInfo* port, GeomID serial, GeomObj *obj,
		const clString&, CrowdMonitor* lock);
    void delObj(PortInfo* port, GeomID serial);
    void delAll(PortInfo* port);
    void flushPort(int portid);
    void flushViews();
    void addTopRoe(Roe *r);
    void delTopRoe(Roe *r);

    void tcl_command(TCLArgs&, void*);

    // The scene...
    HashTable<int, PortInfo*> portHash;

    // Lighting
    Lighting lighting;

    int process_event(int block);

    int lookup_specific(const clString& key, void*&);
    void insert_specific(const clString& key, void* data);
};

class SalmonMessage : public MessageBase {
public:
    clString rid;
    clString filename;
    SalmonMessage(const clString& rid);
    SalmonMessage(const clString& rid, const clString& filename);
    virtual ~SalmonMessage();
};

#endif

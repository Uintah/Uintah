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
#include <Geom/Lighting.h>
#include <Multitask/ITC.h>

class GeomReply;
class Roe;

struct SceneItem {
    GeomObj* obj;
    clString name;

    SceneItem(GeomObj*, const clString&);
    ~SceneItem();
};

class Salmon : public Module {
    Array1<Roe*> roe;
    int busy_bit;
    Array1<Roe*> topRoe;
    virtual void do_execute();
    virtual int should_execute();

    int max_portno;
    virtual void connection(Module::ConnectionMode, int, int);

public:
    MaterialHandle default_matl;
    friend class Roe;
    Salmon(const clString& id);
    Salmon(const Salmon&, int deep);
    virtual ~Salmon();
    virtual Module* clone(int deep);
    virtual void execute();
    void initPort(Mailbox<GeomReply>*);
    void addObj(int portno, GeomID serial, GeomObj *obj,
		const clString&);
    void delObj(int portno, GeomID serial);
    void delAll(int portno);
    void flushViews();
    void addTopRoe(Roe *r);
    void delTopRoe(Roe *r);

    void tcl_command(TCLArgs&, void*);

    // The scene...
    HashTable<int, HashTable<int, SceneItem*>*> portHash;

    // Lighting
    Lighting lighting;
};

class RedrawMessage : public MessageBase {
public:
    clString rid;
    RedrawMessage(const clString& rid);
    virtual ~RedrawMessage();
};

#endif

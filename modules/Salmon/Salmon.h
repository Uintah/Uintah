/*
 *  Salmon.h: The Geometry Viewer!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Salmon_h
#define SCI_project_module_Salmon_h

#include <Module.h>
#include <GeometryPort.h>
#include <Classlib/Array1.h>
#include <Multitask/ITC.h>
#include <Classlib/HashTable.h>

class CallbackData;
class GeomObj;
class GeometryIPort;
class MaterialProp;
class PopupMenuC;
class PushButtonC;
class Roe;
class XQColor;

class Salmon : public Module {
    Array1<Roe*> topRoe;
    virtual void do_execute();
    virtual void create_interface();
    virtual int should_execute();
    virtual void reconfigure_iports();
    virtual void reconfigure_oports();
    DrawingAreaC* drawing_a;
    void redraw_widget(CallbackData*, void*);
    void widget_button(CallbackData*, void*);
    void move_widget(CallbackData*, void*);
    void post_menu(CallbackData*, void*);
    void destroy(CallbackData*, void*);
    void popup_help(CallbackData*, void*);
    XQColor* bgcolor;
    XQColor* fgcolor;
    XQColor* top_shadow;
    XQColor* bottom_shadow;

    // User Interface stuff...
    PushButtonC* btn;
    GC gc;
    int widget_ytitle;
    int last_x, last_y;
    PopupMenuC* popup_menu;
    int width, height;
    int title_width;
    int compute_width();
    int title_left;
    int need_reconfig;

    int max_portno;
    virtual void connection(Module::ConnectionMode, int, int);

    //gotta store the geometry!
    HashTable<int, HashTable<int, GeomObj*>*> portHash;

    MaterialProp* default_matl;
public:
    friend class Roe;
    Salmon();
    Salmon(const Salmon&, int deep);
    virtual ~Salmon();
    virtual Module* clone(int deep);
    void initPort(Mailbox<int>*);
    void addObj(int portno, GeomID serial, GeomObj *obj,
		const clString&);
    void delObj(int portno, GeomID serial);
    void delAll(int portno);
    void flushViews();
    void addTopRoe(Roe *r);
    void delTopRoe(Roe *r);
    void spawnIndCB(CallbackData*, void*);
    void printFamilyTree();
};

#endif

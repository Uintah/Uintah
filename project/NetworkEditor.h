
/*
 *  NetworkEditor.h: Interface to Network Editor class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_NetworkEditor_h
#define SCI_project_NetworkEditor_h 1

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <ModuleList.h>
#include <MessageBase.h>
#include <WrapperLite.h>
#include <X11/Xlib.h>
class ApplicationShellC;
class CallbackData;
class ColorManager;
class Connection;
class Datatype;
class DrawingAreaC;
class ListC;
class MemStats;
class MessageBase;
class Module;
class Network;
class TextC;
class XFont;

class NetworkEditor : public Task {
    Network* net;
    void do_scheduling();
    int first_schedule;
    MemStats* memstats;
    void popup_memstats(CallbackData*, void*);
    void quit(CallbackData*, void*);

    void redraw(CallbackData*, void*);
    void rightmouse(CallbackData*, void*);
    void draw_temp_portlines();

    ListC* list1;
    ListC* list2;
    ListC* list3;
    void list1_cb(CallbackData*, void*);
    void list2_cb(CallbackData*, void*);
    void list3_cb(CallbackData*, void*);

    TextC* text;

    void update_list(ModuleCategory*);
    void update_list(ModuleSubCategory*);
    ModuleDB* current_db;
    ModuleCategory* current_cat;
    ModuleSubCategory* current_subcat;

    int making_connection;
    Connection* conn_in_progress;
    Module* from_module;
    int from_which;
    int from_oport;
    Module* to_module;
    int to_which;
    void update_to(int x, int y);
    void closeness(Module*, int, int, Module*&, int&, int&);
public:
    // The user interface..
    ApplicationShellC* window;
    DrawingAreaC* drawing_a;
    Display* display;
    ColorManager* color_manager;
    XFont* name_font;
    XFont* time_font;
    Mailbox<MessageBase*> mailbox;

    NetworkEditor(Network*, Display*, ColorManager*);
    ~NetworkEditor();

    void connection_cb(CallbackData*, void*);
    int check_cancel();

    void add_text(const clString&);
private:
    virtual int body(int);
    void main_loop();
};

class Scheduler_Module_Message : public MessageBase {
public:
    Scheduler_Module_Message();
    virtual ~Scheduler_Module_Message();
};

class Module_Scheduler_Message : public MessageBase {
public:
    Module_Scheduler_Message();
    virtual ~Module_Scheduler_Message();
};

#endif

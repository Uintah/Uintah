
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
#include <MessageBase.h>
#include <WrapperLite.h>
#include <X11/Xlib.h>
class ApplicationShellC;
class CallbackData;
class ColorManager;
class Connection;
class Datatype;
class DrawingAreaC;
class MemStats;
class MessageBase;
class Module;
class Network;

class NetworkEditor : public Task {
    Network* net;
    void do_scheduling();
    int first_schedule;
    MemStats* memstats;
    void popup_memstats(CallbackData*, void*);
public:
    // The user interface..
    ApplicationShellC* window;
    DrawingAreaC* drawing_a;
    Display* display;
    ColorManager* color_manager;
    XFontStruct* name_font;
    XFontStruct* time_font;
    Mailbox<MessageBase*> mailbox;

    NetworkEditor(Network*, Display*, ColorManager*);
    ~NetworkEditor();
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


/*
 *  UserModule.h: Base class for defined modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_UserModule_h
#define SCI_project_UserModule_h 1

#include <Module.h>
#include <Classlib/Timer.h>
#include <X11/Xlib.h>
class CallbackData;
class DrawingAreaC;
class PopupMenuC;
class PushButtonC;
class MUI_window;
class MUI_widget;
class MUI_onoff_switch;
class InData;
class OutData;
class XQColor;

class UserModule : public Module {
    MUI_window* window;
    int popup_on_create;
    int old_gwidth;
    WallClockTimer timer;
    PushButtonC* btn;
    int need_reconfig;
    int last_x, last_y;
    int compute_width();
public:
    UserModule(const clString& name, SchedClass);
    UserModule(const UserModule&, int deep);
    virtual ~UserModule();

    // User Interface Manipulations
    void remove_ui(MUI_widget*);
    void add_ui(MUI_widget*);
    void reconfigure_ui();

    // Misc stuff for module writers
    void error(const clString&);
    void update_progress(double);
    void update_progress(int, int);

    // Callbacks...
    virtual void create_interface();
    void redraw_widget(CallbackData*, void*);
    void widget_button(CallbackData*, void*);
    void draw_button(Display*, Window, GC, int);
    void move_widget(CallbackData*, void*);
    void post_menu(CallbackData*, void*);
    void popup_help(CallbackData*, void*);
    void destroy(CallbackData*, void*);
    void interrupt(CallbackData*, void*);

    void update_module(int);
    virtual void do_execute();
    virtual void execute()=0;
    virtual int should_execute();
    virtual void mui_callback(void*, int);

    // Our interface...
    virtual void get_iport_coords(int, int&, int&);
    virtual void get_oport_coords(int, int&, int&);
    virtual void reconfigure_iports();
    virtual void reconfigure_oports();
    GC gc;
    DrawingAreaC* drawing_a;
    PopupMenuC* popup_menu;
    XQColor* bgcolor;
    XQColor* fgcolor;
    XQColor* top_shadow;
    XQColor* bottom_shadow;
    XQColor* select_color;
    XQColor* executing_color;
    XQColor* executing_color_top;
    XQColor* executing_color_bot;
    XQColor* completed_color;
    XQColor* completed_color_top;
    XQColor* completed_color_bot;
    int widget_ytitle;
    int widget_ygraphtop;
    int widget_ytime;
    int widget_ygraphbot;
    int widget_xgraphleft;
    int widget_xgraphright;
    int title_left;
    int title_width;
};

#endif /* SCI_project_UserModule_h */


/*
 *  Dialbox.h: Dialbox manager thread...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_project_Dialbox_h
#define sci_project_Dialbox_h 1

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <DBContext.h>
#include <MessageBase.h>
#include <X11/Xlib.h>
class CallbackData;
class ColorManager;
class DialogShellC;
class DrawingAreaC;
class XQColor;

struct DialMsg : public MessageBase {
    DBContext* context;
    DialMsg(DBContext*);
    int which;
    int info;
    DialMsg(int, int);
    ~DialMsg();
};

class Dialbox : public Task {
    ColorManager* color_manager;
    DBContext* context;
    Mailbox<MessageBase*> mailbox;

    Display* dpy;
    GC gc;
    XFontStruct* font;
    XQColor* bgcolor;
    XQColor* top_shadow;
    XQColor* bottom_shadow;
    XQColor* fgcolor;
    XQColor* inset_color;

    void popup_ui();
    DialogShellC* window;
    DrawingAreaC* main_da;
    DrawingAreaC* title_da;
    DrawingAreaC* dial_da[8];
    void redraw_title(CallbackData*, void*);
    void redraw_dial(CallbackData*, void*);
public:
    Dialbox(ColorManager*);
    virtual ~Dialbox();

    virtual int body(int);

    static void attach_dials(DBContext*);
    static int get_event_type();
    static void handle_event(void*);
};

#endif


/*
 *  ThreadStats.h: Thread information visualizer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ThreadStats_h
#define SCI_project_ThreadStats_h 1

#include <X11/Xlib.h>

class CallbackData;
class DialogShellC;
class DrawingAreaC;
class EncapsulatorC;
class NetworkEditor;
class PushButtonC;
class TaskInfo;
class XFont;
class XQColor;

class ThreadStats {
    NetworkEditor* netedit;
    DialogShellC* dialog;
    DrawingAreaC** title_da;
    DrawingAreaC** graph_da;
    PushButtonC** dbx_btn;
    PushButtonC** core_btn;

    XFont* stats_font;
    Display* dpy;
    GC gc;
    void write_text_line(EncapsulatorC*, int, int&, char*);
    int ascent, descent;
    int textwidth;
    int line_height;
    XQColor* fgcolor;
    XQColor* stack_used_color;
    XQColor* stack_free_color;

    int sizewidth;
    int graphwidth;
    TaskInfo* info;
    int maxstacksize;
    void redraw_title(CallbackData*, void*);
    void redraw_graph(CallbackData*, void*);
    void do_dbx(CallbackData*, void*);
    void do_coredump(CallbackData*, void*);
    void timer(CallbackData*, void*);
public:
    ThreadStats(NetworkEditor* netedit);
    ~ThreadStats();
    void popup();
};

#endif


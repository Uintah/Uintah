
/*
 *  MemStats.h: Interface to memory stats...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MemStats_h
#define SCI_project_MemStats_h 1

#include <X11/Xlib.h>
class CallbackData;
class DialogShellC;
class DrawingAreaC;
class NetworkEditor;
class XQColor;

class MemStats {
    NetworkEditor* netedit;
    DialogShellC* dialog;
    DrawingAreaC* drawing_a;

    XFontStruct* stats_font;
    Drawable win;
    Display* dpy;
    GC gc;
    void write_text_line(int, int&, char*);
    XQColor* fgcolor;
    XQColor* unfreed_color;
    XQColor* inlist_color;

    int textwidth;
    int graphwidth;
    int width, height;
    int line_height;
    int ascent, descent;
    int nbins;
    int nnz;
    int* lines;
    int* old_reqd;
    int* old_deld;
    int* old_inlist;
    int* old_ssize;
    int* old_lsize;
    long old_nnew, old_snew, old_nfillbin, old_ndelete;
    long old_sdelete, old_nsbrk, old_ssbrk;
    void redraw(CallbackData*, void*);
    void timer(CallbackData*, void*);

    void redraw_bin(int bin, int line, int cflag);
    void redraw_globals(int cflag);
public:
    MemStats(NetworkEditor* netedit);
    ~MemStats();
    void popup();
};

#endif


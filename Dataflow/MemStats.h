
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

#include <TCL.h>

class MemStats : public TCL {
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

    int redraw_globals;
public:
    MemStats();
    ~MemStats();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

#endif


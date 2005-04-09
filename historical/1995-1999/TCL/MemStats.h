
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

#include <TCL/TCL.h>
struct Allocator;
#include <unistd.h>

class MemStats : public TCL {
    Allocator* a;
    int textwidth;
    int graphwidth;
    int width, height;
    int nbins;
    int nnz;
    int* lines;
    size_t* old_reqd;
    size_t* old_deld;
    size_t* old_inlist;
    size_t* old_ssize;
    size_t* old_lsize;
    size_t old_nalloc, old_sizealloc, old_nfree, old_sizefree;
    size_t old_nfillbin;
    size_t old_nmmap, old_sizemmap, old_nmunmap, old_sizemunmap;
    size_t old_highwater_alloc, old_highwater_mmap;
    size_t old_nlonglocks, old_nnaps, old_bytes_overhead;
    size_t old_bytes_free, old_bytes_fragmented, old_bytes_inuse;
    size_t old_bytes_inhunks;

    int redraw_globals;
public:
    MemStats();
    ~MemStats();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

#endif


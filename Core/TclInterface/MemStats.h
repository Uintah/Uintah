
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

#include <TclInterface/TCL.h>
#include <unistd.h>

namespace SCICore {
  namespace Malloc {
    struct Allocator;
  }
}

namespace SCICore {
namespace TclInterface {

using SCICore::Malloc::Allocator;

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

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:15  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:23  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:33  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif


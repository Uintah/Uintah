/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/GuiInterface/GuiCallback.h>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace SCIRun {

struct Allocator;
  class GuiInterface;

class SCICORESHARE MemStats : public GuiCallback {
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

    void init_tcl(GuiInterface* gui);
    virtual void tcl_command(GuiArgs&, void*);
};

} // End namespace SCIRun


#endif


/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

class MemStats : public GuiCallback {
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


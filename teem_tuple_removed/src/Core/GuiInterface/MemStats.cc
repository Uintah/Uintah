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
 *  MemStats.cc: Interface to memory stats...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/MemStats.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <stdio.h>

namespace SCIRun {


MemStats::MemStats()
{
#ifndef _WIN32
    a=DefaultAllocator();
    if(a){
	nbins=GetNbins(a);

	GetGlobalStats(a, old_nalloc, old_sizealloc, old_nfree, old_sizefree,
		       old_nfillbin,
		       old_nmmap, old_sizemmap, old_nmunmap, old_sizemunmap,
		       old_highwater_alloc, old_highwater_mmap,
		       old_bytes_overhead,
		       old_bytes_free, old_bytes_fragmented, old_bytes_inuse,
		       old_bytes_inhunks);

	old_ssize=scinew size_t[nbins];
	old_lsize=scinew size_t[nbins];
	old_reqd=scinew size_t[nbins];
	old_deld=scinew size_t[nbins];
	old_inlist=scinew size_t[nbins];
	lines=scinew int[nbins];
	nnz=0;
	for(int i=0;i<nbins;i++){
	    GetBinStats(a, i, old_ssize[i], old_lsize[i],
			old_reqd[i], old_deld[i], old_inlist[i]);
	    if(old_inlist[i] || old_reqd[i] || old_deld[i]){
		lines[i]=nnz++;
	    }
	}
    }
#endif
}

MemStats::~MemStats()
{
}
void MemStats::init_tcl(GuiInterface* gui)
{
  gui->add_command("memstats", this, 0);
}

void MemStats::tcl_command(GuiArgs& args, void*)
{
#ifndef _WIN32
    if(args.count() < 2){
	args.error("memstats needs a minor command");
	return;
    }
    if(args[1] == "binchange"){
	if(redraw_globals){
	    // Check bins too...
	    int old_nnz=nnz;
	    nnz=0;
	    int changed=0;
	    int i;
	    for(i=0;i<nbins;i++){
		size_t ssize, lsize, n_reqd, n_deld, n_inlist;
		GetBinStats(a, i, ssize, lsize, n_reqd,
			    n_deld, n_inlist);
		if(n_inlist || n_reqd || n_deld){
		    if(lines[i] != nnz){
			lines[i]=nnz;
			changed=1;
		    }
		    nnz++;
		}
		if((changed && (n_inlist || n_reqd || n_deld))
		   || n_reqd != old_reqd[i]
		   || n_deld != old_deld[i]
		   || n_inlist != old_inlist[i]){
		    old_inlist[i]=n_inlist;
		    old_reqd[i]=n_reqd;
		    old_deld[i]=n_deld;
		    args.append_element(to_string(i));
		}
	    }
	    for(i=nnz;i<old_nnz;i++){
		args.append_element(to_string(-i));
	    }
	}
    } else if(args[1] == "nbins") {
	args.result(to_string(nbins));
    } else if(args[1] == "bin"){
	if(args.count() != 3){
	    args.error("No bin requested!");
	    return;
	}
	int bin;
	if(!string_to_int(args[2], bin)) {
	    args.error("Error parsing bin argument");
	    return;
	}
	char buf[100];
	sprintf(buf, "%d|%d|%d|%7d-%7d : %9d%9d %7d%7d",
		lines[bin], old_inlist[bin], old_reqd[bin]-old_deld[bin],
		old_ssize[bin], old_lsize[bin],
		old_reqd[bin], old_deld[bin], old_inlist[bin],
		old_reqd[bin]-old_deld[bin]);
	args.result(buf);
    } else if(args[1] == "globalstats"){
	size_t nalloc, sizealloc, nfree, sizefree, nmmap, sizemmap;
	size_t nfillbin;
	size_t nmunmap, sizemunmap, highwater_alloc, highwater_mmap;
	size_t bytes_overhead, bytes_free;
	size_t bytes_fragmented, bytes_inuse, bytes_inhunks;
	GetGlobalStats(a, nalloc, sizealloc, nfree, sizefree,
		       nfillbin,
		       nmmap, sizemmap, nmunmap, sizemunmap,
		       highwater_alloc, highwater_mmap,
		       bytes_overhead,
		       bytes_free, bytes_fragmented, bytes_inuse,
		       bytes_inhunks);
	if(nalloc != old_nalloc || nfree != old_nfree){
	    redraw_globals=1;
	    old_nalloc=nalloc;
	    old_sizealloc=sizealloc;
	    old_nfree=nfree;
	    old_sizefree=sizefree;
	    old_nfillbin=nfillbin;
	    old_nmmap=nmmap;
	    old_sizemmap=sizemmap;
	    old_nmunmap=nmunmap;
	    old_sizemunmap=sizemunmap;
	    old_highwater_alloc=highwater_alloc;
	    old_highwater_mmap=highwater_mmap;
	    old_bytes_overhead=bytes_overhead;
	    old_bytes_free=bytes_free;
	    old_bytes_fragmented=bytes_fragmented;
	    old_bytes_inuse=bytes_inuse;
	    old_bytes_inhunks=bytes_inhunks;

	    size_t total=bytes_overhead+bytes_free+bytes_fragmented+bytes_inuse+bytes_inhunks;
	    char buf[1000];
	    sprintf(buf,
		    "Calls to malloc/new: %ld (%ld bytes)\n"
		    "Calls to free/delete: %ld (%ld bytes)\n"
		    "Missing allocations: %ld (%ld bytes)\n"
		    "Allocation highwater mark: %ld bytes\n"
		    "Calls to fillbin: %ld (%.2f%%)\n"
		    "Requests from system: %ld (%ld bytes)\n"
		    "Returned to system: %ld (%ld bytes)\n"
		    "System highwater mark: %ld bytes\n"
		    "\nBreakdown:\n"
		    "Inuse: %d bytes (%.2f%%)\n"
		    "Free: %d bytes (%.2f%%)\n"
		    "Overhead: %d bytes (%.2f%%)\n"
		    "Fragmentation: %d bytes (%.2f%%)\n"
		    "Left in hunks: %d bytes (%.2f%%)\n"
		    "Total: %d bytes (%.2f%%)\n"
		    "\n  ssize   lsize        reqd     deld  inlist  inuse",
		    (long)nalloc, (long)sizealloc,
		    (long)nfree, (long)sizefree,
		    (long)(nalloc-nfree), (long)(sizealloc-sizefree),
		    (long)highwater_alloc,
		    (long)nfillbin, 100.*(double)nfillbin/(double)nalloc,
		    (long)nmmap, (long)sizemmap,
		    (long)nmunmap, (long)sizemunmap,
		    (long)highwater_mmap,
		    bytes_inuse, 100.*(double)bytes_inuse/(double)total,
		    bytes_free, 100.*(double)bytes_free/(double)total,
		    bytes_overhead, 100.*(double)bytes_overhead/(double)total,
		    bytes_fragmented, 100.*(double)bytes_fragmented/(double)total,
		    bytes_inhunks, 100.*(double)bytes_inhunks/(double)total,
		    total, 100.);
	    args.result(buf);
	} else {
	    redraw_globals=0;
	    args.result("");
	}
    } else if(args[1] == "audit"){
	AuditAllocator(a);
	fprintf(stderr, "Memory audit OK\n");
    } else if(args[1] == "dump"){
	DumpAllocator(a);
    } else {
	args.error("Unknown minor command for memstats");
    }
#endif
}

} // End namespace SCIRun


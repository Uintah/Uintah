
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

#include <MemStats.h>
#include <Malloc/New.h>
#include <iostream.h>
#include <stdio.h>

MemStats::MemStats()
{
    MemoryManager::get_global_stats(old_nnew, old_snew, old_nfillbin,
				    old_ndelete, old_sdelete,
				    old_nsbrk, old_ssbrk);
    nbins=MemoryManager::get_nbins();
    old_ssize=new int[nbins];
    old_lsize=new int[nbins];
    old_reqd=new int[nbins];
    old_deld=new int[nbins];
    old_inlist=new int[nbins];
    lines=new int[nbins];
    nnz=0;
    for(int i=0;i<nbins;i++){
	MemoryManager::get_binstats(i, old_ssize[i], old_lsize[i],
				    old_reqd[i], old_deld[i], old_inlist[i]);
	if(old_inlist[i] || old_reqd[i] || old_deld[i]){
	    lines[i]=nnz++;
	}
    }

}

MemStats::~MemStats()
{
}
void MemStats::init_tcl()
{
    TCL::add_command("memstats", this, 0);
}

void MemStats::tcl_command(TCLArgs& args, void*)
{
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
	    for(int i=0;i<nbins;i++){
		int ssize, lsize, n_reqd, n_deld, n_inlist;
		MemoryManager::get_binstats(i, ssize, lsize, n_reqd,
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
	if(!args[2].get_int(bin)){
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
	long nnew, snew, nfillbin, ndelete, sdelete, nsbrk, ssbrk;
	MemoryManager::get_global_stats(nnew, snew, nfillbin, ndelete, sdelete,
					nsbrk, ssbrk);
	if(nnew != old_nnew || ndelete != old_ndelete){
	    redraw_globals=1;
	    old_nnew=nnew;
	    old_snew=snew;
	    old_nfillbin=nfillbin;
	    old_ndelete=ndelete;
	    old_sdelete=sdelete;
	    old_nsbrk=nsbrk;
	    old_ssbrk=ssbrk;

	    char buf[500];
	    sprintf(buf, "Calls to malloc/new: %d (%d bytes)\nCalls to free/delete: %d (%d bytes)\nCalls to fillbin: %d (%.2f%%)\nRequests from system: %d (%d bytes)\nMissing allocations: %d (%d bytes)\n\n  ssize   lsize        reqd     deld  inlist  inuse", old_nnew, old_snew, old_ndelete, old_sdelete, old_nfillbin, 100.*(double)old_nfillbin/(double)old_nnew, old_nsbrk, old_ssbrk, old_nnew-old_ndelete, old_snew-old_sdelete);
	    args.result(buf);
	} else {
	    redraw_globals=0;
	    args.result("");
	}
    } else {
	args.error("Unknown minor command for memstats");
    }
}

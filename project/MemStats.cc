
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

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <MemStats.h>
#include <MotifCallback.h>
#include <NetworkEditor.h>
#include <XQColor.h>
#include <Malloc/New.h>
#include <MtXEventLoop.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <iostream.h>
#include <stdio.h>
extern MtXEventLoop* evl;

#define MEMSTATS_FONT "screen14"
#define MEMSTATS_FGCOLOR "black"
#define MEMSTATS_INLISTCOLOR "blue"
#define MEMSTATS_UNFREEDCOLOR "red"
#define MEMSTATS_UPDATE_MS 500 // 2X per second...

MemStats::MemStats(NetworkEditor* netedit)
: netedit(netedit)
{
    evl->lock();
    dialog=new DialogShellC;
    dialog->Create("Memory Stats", "Memory Stats", evl->get_display());
    drawing_a=new DrawingAreaC;
    textwidth=355;
    graphwidth=200;
    drawing_a->SetWidth(textwidth+graphwidth);
    drawing_a->SetHeight(830);
    drawing_a->SetShadowThickness(0);
    drawing_a->SetResizePolicy(XmRESIZE_NONE);
    // Add redraw callback...
    new MotifCallback<MemStats>FIXCB(drawing_a, XmNexposeCallback,
				     &netedit->mailbox, this,
				     &MemStats::redraw, 0, 0);
    drawing_a->Create(*dialog, "memory_stats");

    dpy=evl->get_display();
    if( (stats_font = XLoadQueryFont(dpy, MEMSTATS_FONT)) == 0){
	cerr << "Error loading font: " << stats_font << endl;
	exit(-1);
    }
    win=XtWindow(*drawing_a);
    gc=XCreateGC(dpy, win, 0, 0);
    fgcolor=new XQColor(netedit->color_manager, MEMSTATS_FGCOLOR);
    unfreed_color=new XQColor(netedit->color_manager, MEMSTATS_UNFREEDCOLOR);
    inlist_color=new XQColor(netedit->color_manager, MEMSTATS_INLISTCOLOR);

    // Set up junk...
    XSetFont(dpy, gc, stats_font->fid);
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
    XCharStruct dim;
    int dir;
    if(!XTextExtents(stats_font, "Xpq", 3, &dir, &ascent, &descent,
		     &dim)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    line_height=ascent+descent-1;


    // Create the timer callback...
    new MotifCallback<MemStats>FIXCB(MEMSTATS_UPDATE_MS, MEMSTATS_UPDATE_MS,
				     &netedit->mailbox,
				     this, &MemStats::timer, 0, 0);
    evl->unlock();
}

MemStats::~MemStats()
{
    delete drawing_a;
    delete dialog;
}

void MemStats::write_text_line(int x, int& y, char* text)
{
    Drawable win=XtWindow(*drawing_a);
    y+=ascent;
    int textlen=strlen(text);
    XDrawString(dpy, win, gc, x, y, text, textlen);
    y+=descent;
}

void MemStats::redraw(CallbackData*, void*)
{
    evl->lock();
    int x=2;
    int y=2;
    XClearWindow(dpy, win);
    Dimension w, h;
    drawing_a->GetWidth(&w);
    drawing_a->GetWidth(&h);
    drawing_a->GetValues();
    width=w;
    height=h;
    char buf[1000];
    sprintf(buf, " ssize  lsize        reqd     deld  inlist  inuse");
    XSetForeground(dpy, gc, fgcolor->pixel());
    write_text_line(x,y,buf);
    for(int i=0;i<nbins;i++){
	if(old_inlist[i] || old_reqd[i] || old_deld[i]){
	    redraw_bin(i, lines[i], 0);
	}
    }
    redraw_globals(0);
    evl->unlock();
}

void MemStats::redraw_bin(int bin, int line, int cflag)
{
    int x=2;
    int y=2+(line+1)*line_height;
    if(cflag){
	XClearArea(dpy, win, x, y, width-x, line_height+1, False);
    }
    char buf[1000];
    sprintf(buf, "%6d-%6d : %9d%9d %7d%7d", old_ssize[bin], old_lsize[bin],
	    old_reqd[bin], old_deld[bin], old_inlist[bin],
	    old_reqd[bin]-old_deld[bin]);
    XSetForeground(dpy, gc, fgcolor->pixel());
    write_text_line(x,y,buf);

    y=2+(line+1)*line_height;
    int unfreed=old_reqd[bin]-old_deld[bin];
    int inlist=old_inlist[bin];
    int total=inlist+unfreed;
    int scale=100;
    while(total >= scale)scale*=10;
    int left=textwidth+5;
    int gwidth=graphwidth-5-2;
    int ilwidth=gwidth*inlist/scale;
    int ufwidth=gwidth*unfreed/scale;
    XSetForeground(dpy, gc, inlist_color->pixel());
    XFillRectangle(dpy, win, gc, left, y+1, ilwidth, line_height-1);
    XSetForeground(dpy, gc, unfreed_color->pixel());
    XFillRectangle(dpy, win, gc, left+ilwidth, y+1, ufwidth, line_height-1);
    if(scale != 100){
	XSetForeground(dpy, gc, fgcolor->pixel());
	x=left+ilwidth+ufwidth+2;
	sprintf(buf, "*%d", scale/100);
	write_text_line(x,y,buf);
    }
}

void MemStats::redraw_globals(int cflag)
{
    int x=2;
    int y=2+(nnz+1)*line_height;
    if(cflag){
	XClearArea(dpy, win, x, y, width-x, line_height*7, False);
    }
    XSetForeground(dpy, gc, fgcolor->pixel());
    char buf[1000];
    sprintf(buf, "Calls to malloc/new: %d (%d bytes)", old_nnew, old_snew);
    write_text_line(x, y, buf);
    sprintf(buf, "Calls to free/delete: %d (%d bytes)", old_ndelete, old_sdelete);
    write_text_line(x, y, buf);
    sprintf(buf, "Calls to fillbin: %d (%.2f%%)",
	    old_nfillbin, 100.*(double)old_nfillbin/(double)old_nnew);
    write_text_line(x, y, buf);
    sprintf(buf, "Requests from system: %d (%d bytes)", old_nsbrk, old_ssbrk);
    write_text_line(x, y, buf);
    sprintf(buf, "Missing allocations: %d (%d bytes)",
	    old_nnew-old_ndelete, old_snew-old_sdelete);
    write_text_line(x, y, buf);
}

void MemStats::timer(CallbackData*, void*)
{
    long nnew, snew, nfillbin, ndelete, sdelete, nsbrk, ssbrk;
    MemoryManager::get_global_stats(nnew, snew, nfillbin, ndelete, sdelete,
				    nsbrk, ssbrk);
    if(nnew != old_nnew || ndelete != old_ndelete){
	old_nnew=nnew;
	old_snew=snew;
	old_nfillbin=nfillbin;
	old_ndelete=ndelete;
	old_sdelete=sdelete;
	old_nsbrk=nsbrk;
	old_ssbrk=ssbrk;
	evl->lock();
	redraw_globals(1);

	// Check bins too...
	int locked=0;
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
		redraw_bin(i, lines[i], 1);
	    }
	}
	if(changed){
	    // Redraw globals...
	    redraw_globals(1);
	}
	evl->unlock();
    }
}

void MemStats::popup()
{
    evl->lock();
    XtPopup(*dialog, XtGrabNone);
    evl->unlock();
}



/*
 *  ThreadStats.cc: Interface to memory stats...
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

#include <ThreadStats.h>
#include <MotifCallback.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <XFont.h>
#include <XQColor.h>
#include <MtXEventLoop.h>
#include <Mt/DialogShell.h>
#include <Mt/DrawingArea.h>
#include <Mt/PushButton.h>
#include <Mt/RowColumn.h>
#include <iostream.h>
#include <stdio.h>
extern MtXEventLoop* evl;

#define MAXTHREADS 12
#define THREADSTATS_FONTSIZE 14
#define THREADSTATS_FONTFACE XFont::Medium
#define THREADSTATS_FGCOLOR "black"
#define THREADSTATS_USEDCOLOR "red"
#define THREADSTATS_FREECOLOR "blue"
#define THREADSTATS_UPDATE_MS 500 // 2X per second...

ThreadStats::ThreadStats(NetworkEditor* netedit)
: netedit(netedit)
{
    evl->lock();
    stats_font=new XFont(THREADSTATS_FONTSIZE, THREADSTATS_FONTFACE, 1);
    dialog=new DialogShellC;
    dialog->Create("Thread Stats", "Thread Stats", evl->get_display());
    int dir;
    XCharStruct dim;
    char* tname="This is a long thread name (pid 9999)";
    if(!XTextExtents(stats_font->font, tname, strlen(tname), &dir, &ascent, &descent,
		     &dim)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    textwidth=dim.width;
    line_height=ascent+descent-1;
    char* ss="128k";
    if(!XTextExtents(stats_font->font, ss, strlen(ss), &dir, &ascent, &descent,
		     &dim)){
	cerr << "XTextExtents failed...\n";
	exit(-1);
    }
    sizewidth=dim.width;
    graphwidth=textwidth/2+2+sizewidth;
    RowColumnC* rc=new RowColumnC;
    rc->SetOrientation(XmVERTICAL);
    rc->Create(*dialog, "rc");
    title_da=new DrawingAreaC*[MAXTHREADS];
    graph_da=new DrawingAreaC*[MAXTHREADS];
    dbx_btn=new PushButtonC*[MAXTHREADS];
    core_btn=new PushButtonC*[MAXTHREADS];
    for(int i=0;i<MAXTHREADS;i++){
	RowColumnC* rc2=new RowColumnC;
	rc2->SetOrientation(XmHORIZONTAL);
	rc2->Create(*rc, "rc");
	PushButtonC* pb1=new PushButtonC;
	new MotifCallback<ThreadStats>FIXCB(pb1, XmNactivateCallback,
					    &netedit->mailbox, this,
					    &ThreadStats::do_dbx,
					    (void*)i, 0);
	pb1->Create(*rc2, "DBX");
	PushButtonC* pb2=new PushButtonC;
	new MotifCallback<ThreadStats>FIXCB(pb2, XmNactivateCallback,
					    &netedit->mailbox, this,
					    &ThreadStats::do_coredump,
					    (void*)i, 0);
	pb2->Create(*rc2, "Core dump");
	dbx_btn[i]=pb1;
	core_btn[i]=pb2;
	DrawingAreaC* da1=new DrawingAreaC;
	da1->SetWidth(textwidth);
	da1->SetHeight(2*line_height);
	da1->SetResizePolicy(XmRESIZE_NONE);
	da1->SetShadowThickness(0);
	new MotifCallback<ThreadStats>FIXCB(da1, XmNexposeCallback,
					    &netedit->mailbox, this,
					    &ThreadStats::redraw_title,
					    (void*)i, 0);
	da1->Create(*rc2, "threadname");
	DrawingAreaC* da2=new DrawingAreaC;
	da2->SetWidth(graphwidth);
	da2->SetHeight(2*line_height);
	da2->SetResizePolicy(XmRESIZE_NONE);
	da2->SetShadowThickness(0);
	new MotifCallback<ThreadStats>FIXCB(da2, XmNexposeCallback,
					    &netedit->mailbox, this,
					    &ThreadStats::redraw_graph,
					    (void*)i, 0);
	da2->Create(*rc2, "stackgraph");
	title_da[i]=da1;
	graph_da[i]=da2;
    }
    
    dpy=evl->get_display();
    Window win=XtWindow(*rc);
    gc=XCreateGC(dpy, win, 0, 0);
    fgcolor=new XQColor(netedit->color_manager, THREADSTATS_FGCOLOR);
    stack_used_color=new XQColor(netedit->color_manager, THREADSTATS_USEDCOLOR);
    stack_free_color=new XQColor(netedit->color_manager, THREADSTATS_FREECOLOR);

    // Set up junk...
    XSetFont(dpy, gc, stats_font->font->fid);
    info=TaskManager::get_taskinfo();
    maxstacksize=0;
    for(i=0;i<info->ntasks;i++)
	if(info->tinfo[i].stacksize > maxstacksize)
	    maxstacksize=info->tinfo[i].stacksize;
    for(i=info->ntasks;i<MAXTHREADS;i++){
	dbx_btn[i]->SetSensitive(False);
	dbx_btn[i]->SetValues();
	core_btn[i]->SetSensitive(False);
	core_btn[i]->SetValues();
    }
    // Create the timer callback...
    new MotifCallback<ThreadStats>FIXCB(THREADSTATS_UPDATE_MS,
					THREADSTATS_UPDATE_MS,
					&netedit->mailbox,
					this, &ThreadStats::timer, 0, 0);
    evl->unlock();
}

ThreadStats::~ThreadStats()
{
}

void ThreadStats::write_text_line(EncapsulatorC* drawing_a,
				  int x, int& y, char* text)
{
    Drawable win=XtWindow(*drawing_a);
    y+=ascent;
    int textlen=strlen(text);
    XDrawString(dpy, win, gc, x, y, text, textlen);
    y+=descent;
}

void ThreadStats::timer(CallbackData*, void*)
{
    TaskInfo* oldinfo=info;
    info=TaskManager::get_taskinfo();
    maxstacksize=0;
    for(int i=0;i<info->ntasks;i++)
	if(info->tinfo[i].stacksize > maxstacksize)
	    maxstacksize=info->tinfo[i].stacksize;
    evl->lock();
    for(i=oldinfo->ntasks;i<info->ntasks;i++){
	dbx_btn[i]->SetSensitive(True);
	dbx_btn[i]->SetValues();
	core_btn[i]->SetSensitive(True);
	core_btn[i]->SetValues();
    }
    for(i=info->ntasks;i<oldinfo->ntasks;i++){
	dbx_btn[i]->SetSensitive(False);
	dbx_btn[i]->SetValues();
	core_btn[i]->SetSensitive(False);
	core_btn[i]->SetValues();
    }
    for(i=0;i<info->ntasks;i++){
	if(i >= oldinfo->ntasks
	   || info->tinfo[i].name != oldinfo->tinfo[i].name
	   || info->tinfo[i].pid != oldinfo->tinfo[i].pid
	   || info->tinfo[i].stacksize != oldinfo->tinfo[i].stacksize
	   || info->tinfo[i].stackused != oldinfo->tinfo[i].stackused){
	    redraw_title(0, (void*)i);
	    redraw_graph(0, (void*)i);
	}
    }
    evl->unlock();
}

void ThreadStats::popup()
{
    evl->lock();
    XtPopup(*dialog, XtGrabNone);
    evl->unlock();
}

void ThreadStats::redraw_title(CallbackData*, void* cbdata)
{
    evl->lock();
    int which=(int)cbdata;
    Window win=XtWindow(*title_da[which]);
    XClearWindow(dpy, win);
    if(which >= info->ntasks){
	evl->unlock();
	return;
    }
    clString name(info->tinfo[which].name);
    int lh=line_height/2;
    char buf[1000];
    sprintf(buf, "%s (pid %d)", name(), info->tinfo[which].pid);
    XSetForeground(dpy, gc, fgcolor->pixel());
    write_text_line(title_da[which], 2, lh, buf);
    evl->unlock();
}

void ThreadStats::redraw_graph(CallbackData*, void* cbdata)
{
    int which=(int)cbdata;
    evl->lock();
    Window win=XtWindow(*graph_da[which]);
    XClearWindow(dpy, win);
    if(which >= info->ntasks){
	evl->unlock();
	return;
    }
    XSetForeground(dpy, gc, stack_used_color->pixel());
    int h=line_height*2;
    int w=graphwidth-sizewidth-2;
    int hw1=w*info->tinfo[which].stackused/maxstacksize;
    int hw2=w*info->tinfo[which].stacksize/maxstacksize;
    XFillRectangle(dpy, win, gc, 0, 0, hw1, h);
    int lh=0;
    char buf[100];
    sprintf(buf, "%dk", info->tinfo[which].stackused/1024);
    write_text_line(graph_da[which], w+2, lh, buf);
    XSetForeground(dpy, gc, stack_free_color->pixel());
    XFillRectangle(dpy, win, gc, hw1, 0, hw2-hw1, h);
    sprintf(buf, "%dk", info->tinfo[which].stacksize/1024);
    write_text_line(graph_da[which], w+2, lh, buf);
    evl->unlock();
}

void ThreadStats::do_dbx(CallbackData*, void* cbdata)
{
    int which=(int)cbdata;
    TaskManager::debug(info->tinfo[which].taskid);
}

void ThreadStats::do_coredump(CallbackData*, void* cbdata)
{
    int which=(int)cbdata;
    TaskManager::coredump(info->tinfo[which].taskid);
}


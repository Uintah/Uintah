
/*
 *  XFont.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <XFont.h>
#include <MtXEventLoop.h>
#include <Classlib/HashTable.h>
#include <Multitask/Task.h>
#include <X11/Xlib.h>
#include <iostream.h>
#include <stdio.h>

extern MtXEventLoop* evl;

static HashTable<XFontIndex, XFontStruct*> fonts;

XFont::XFont(int pointsize, Face face, int monospaced)
{
    XFontIndex index;
    index.pointsize=pointsize;
    index.face=face;
    index.monospaced=monospaced;
    if(!fonts.lookup(index, font)){
	// Allocate it...
	evl->lock();
	// Try lucida first...
	char* ms_str="*";
	if(monospaced)
	    ms_str="m";
	char* pref_name="lucida";
	if(monospaced)
	    pref_name="screen";
	char name[100];
	sprintf(name, "-*-%s-%s-r-*-*-%d-*-*-*-%s-*-iso8859-*",
		pref_name, face==Bold?"bold":"medium", pointsize, ms_str);
	font=XLoadQueryFont(evl->get_display(), name);
	if(!font){
	    sprintf(name, "-*-*-%s-r-*-*-%d-*-*-*-%s-*-iso8859-*",
		    face==Bold?"bold":"medium", pointsize, ms_str);
	    font=XLoadQueryFont(evl->get_display(), name);
	    if(!font){
		cerr << "Error loading font: size=" << pointsize
		    << " face=" << (face==Bold?"bold":"medium") << endl;
		TaskManager::exit_all(-1);
	    }
	}
	fonts.insert(index, font);
	evl->unlock();
    }
}

int XFontIndex::operator==(const XFontIndex& x)
{
    return pointsize==x.pointsize && face==x.face && monospaced==x.monospaced;
}


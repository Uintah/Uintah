
/*
 *  XQColor.cc: Automagically quantizing colors for X
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <XQColor.h>
#include <ColorManager.h>
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
#include <Multitask/ITC.h>
#include <iostream.h>
#include <Xm/Xm.h>

static int db_initialized=0;
static HashTable<clString, RGBColor> XQColor_named_colors;
static Mutex mutex;

static int to_hex(char c)
{
    if(c>'0' && c<='9')
	return c-'0';
    else if(c>'a' && c<='f')
	return c-'a'+10;
    else if(c>'A' && c<='F')
	return c-'A'+10;
    else
	return 0;
}
	

XQColor::XQColor(ColorManager* manager, const char* name)
: manager(manager)
{
    if(!db_initialized){
	mutex.lock();
	if(!db_initialized){
	    initialize_db();
	}
	mutex.unlock();
    }
    if(name[0] == '#'){
	// A number...
	int r=to_hex(name[1])<<4 | to_hex(name[2]);
	int g=to_hex(name[3])<<4 | to_hex(name[4]);
	int b=to_hex(name[5])<<4 | to_hex(name[6]);
	req_color.r=r/255.0;
	req_color.g=g/255.0;
	req_color.b=b/255.0;
    } else {
	if(!XQColor_named_colors.lookup(clString(name), req_color)){
	    cerr << "Error looking up color: " << name << endl;
	    req_color.black();
	}
    }
    register_color();
}

XQColor::XQColor(ColorManager* manager, unsigned int value)
: manager(manager)
{
    double r=double((value>>4)&0xff)/255.0;
    double g=double((value>>2)&0xff)/255.0;
    double b=double((value>>0)&0xff)/255.0;
    req_color=RGBColor(r,g,b);
    register_color();
}

XQColor::XQColor(ColorManager* manager, const RGBColor& c)
: manager(manager), req_color(c)
{
    register_color();
}

XQColor::~XQColor()
{
    unregister_color();
}


// Initialize the color database...
void XQColor::initialize_db()
{
    db_initialized=1;
    // The database is generated automatically from X11's rgb.txt
#include <XQColorDB.h>
}

void XQColor::register_color()
{
    // This will set the pixel value...
    manager->register_color(this, req_color);
}

void XQColor::unregister_color()
{
    manager->unregister_color(this);
}

unsigned long XQColor::pixel()
{
    return pixl;
}

XQColor* XQColor::top_shadow()
{
    XColor bgcolor;
    bgcolor.red=(short)(req_color.r*65535.0);
    bgcolor.green=(short)(req_color.g*65535.0);
    bgcolor.blue=(short)(req_color.b*65535.0);
    XColor fgcolor;
    XColor selectcolor;
    XColor topshadowcolor;
    XColor botshadowcolor;
    void (*color_proc)(XColor*, XColor*, XColor*, XColor*, XColor*);
    color_proc=XmGetColorCalculation();
    (*color_proc)(&bgcolor, &fgcolor, &selectcolor,
		  &topshadowcolor, &botshadowcolor);
    RGBColor c(topshadowcolor.red/65535.0,
	       topshadowcolor.green/65535.0,
	       topshadowcolor.blue/65535.0);
    return new XQColor(manager, c);
}

XQColor* XQColor::bottom_shadow()
{
    XColor bgcolor;
    bgcolor.red=(short)(req_color.r*65535.0);
    bgcolor.green=(short)(req_color.g*65535.0);
    bgcolor.blue=(short)(req_color.b*65535.0);
    XColor fgcolor;
    XColor selectcolor;
    XColor topshadowcolor;
    XColor botshadowcolor;
    void (*color_proc)(XColor*, XColor*, XColor*, XColor*, XColor*);
    color_proc=XmGetColorCalculation();
    (*color_proc)(&bgcolor, &fgcolor, &selectcolor,
		  &topshadowcolor, &botshadowcolor);
    RGBColor c(botshadowcolor.red/65535.0,
	       botshadowcolor.green/65535.0,
	       botshadowcolor.blue/65535.0);
    return new XQColor(manager, c);
}

XQColor* XQColor::fg_color()
{
    XColor bgcolor;
    bgcolor.red=(short)(req_color.r*65535.0);
    bgcolor.green=(short)(req_color.g*65535.0);
    bgcolor.blue=(short)(req_color.b*65535.0);
    XColor fgcolor;
    XColor selectcolor;
    XColor topshadowcolor;
    XColor botshadowcolor;
    void (*color_proc)(XColor*, XColor*, XColor*, XColor*, XColor*);
    color_proc=XmGetColorCalculation();
    (*color_proc)(&bgcolor, &fgcolor, &selectcolor,
		  &topshadowcolor, &botshadowcolor);
    RGBColor c(fgcolor.red/65535.0,
	       fgcolor.green/65535.0,
	       fgcolor.blue/65535.0);
    return new XQColor(manager, c);
}

XQColor* XQColor::select_color()
{
    XColor bgcolor;
    bgcolor.red=(short)(req_color.r*65535.0);
    bgcolor.green=(short)(req_color.g*65535.0);
    bgcolor.blue=(short)(req_color.b*65535.0);
    XColor fgcolor;
    XColor selectcolor;
    XColor topshadowcolor;
    XColor botshadowcolor;
    void (*color_proc)(XColor*, XColor*, XColor*, XColor*, XColor*);
    color_proc=XmGetColorCalculation();
    (*color_proc)(&bgcolor, &fgcolor, &selectcolor,
		  &topshadowcolor, &botshadowcolor);
    RGBColor c(selectcolor.red/65535.0,
	       selectcolor.green/65535.0,
	       selectcolor.blue/65535.0);
    return new XQColor(manager, c);
}

RGBColor::RGBColor()
{
}

RGBColor::RGBColor(double r, double g, double b)
: r(r), g(g), b(b)
{
}

void RGBColor::black()
{
    r=g=b=0;
}


/*
 *  ColorManager.cc: Manage automagically quantized colors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ColorManager.h>
#include <MtXEventLoop.h>
#include <XQColor.h>
#include <Multitask/ITC.h>
#include <iostream.h>
extern MtXEventLoop* evl;

ColorManager::ColorManager(Display* display, Colormap cmap)
: display(display), cmap(cmap)
{
    mutex=new Mutex;
}

ColorManager::~ColorManager()
{
}

void ColorManager::register_color(XQColor* color, const RGBColor& rgb)
{
    // For now, just allocate them a color...
    // Later, we will do more...
    XColor xcolor;
    xcolor.red=(unsigned short)(rgb.r*65535);
    xcolor.green=(unsigned short)(rgb.g*65535);
    xcolor.blue=(unsigned short)(rgb.b*65535);
    xcolor.flags=DoRed|DoGreen|DoBlue;
    evl->lock();
    if(!XAllocColor(display, cmap, &xcolor)){
	cerr << "Error allocating color...\n";
    }
    evl->unlock();
    color->pixl=xcolor.pixel;
}

void ColorManager::unregister_color(XQColor*)
{
}


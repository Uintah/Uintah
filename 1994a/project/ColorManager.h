
/*
 *  ColorManager.h: Manage automagically quantized colors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColorManager_h
#define SCI_project_ColorManager_h 1

#include <X11/Xlib.h>
class XQColor;
class RGBColor;
class Mutex;

class ColorManager {
    Display* display;
    Colormap cmap;
    Mutex* mutex;
public:
    ColorManager(Display*, Colormap);
    ~ColorManager();

    void register_color(XQColor*, const RGBColor&);
    void unregister_color(XQColor*);
};

#endif

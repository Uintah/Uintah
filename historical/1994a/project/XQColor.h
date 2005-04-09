
/*
 *  XQColor.h: Automagically quantizing colors for X
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_XQColor_h
#define SCI_project_XQColor_h 1

class ColorManager;

class RGBColor {
public:
    double r,g,b;
    RGBColor();
    RGBColor(double, double, double);
    void black();
};

class HSVColor : public RGBColor {
public:
    HSVColor(double, double, double);
};

class XQColor {
    ColorManager* manager;
    RGBColor req_color;
    void register_color();
    void unregister_color();
    void initialize_db();
protected:
    friend class ColorManager;
    unsigned long pixl;
public:
    XQColor(ColorManager*, const char* name);
    XQColor(ColorManager*, unsigned int value);
    XQColor(ColorManager*, const RGBColor& c);
    ~XQColor();

    XQColor* top_shadow();
    XQColor* bottom_shadow();
    XQColor* fg_color();
    XQColor* select_color();

    unsigned long pixel();
};

#endif


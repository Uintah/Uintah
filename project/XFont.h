
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

#ifndef SCI_project_XFont_h
#define SCI_proejct_XFont_h 1

#include <X11/Xlib.h>

struct XFont {
    enum Face {
	Bold,
	Medium,
    };
    XFont(int pointsize, Face face);
    XFontStruct* font;
};

struct XFontIndex {
    int pointsize;
    XFont::Face face;
    int operator==(const XFontIndex&);
};

inline int Hash(const XFontIndex& k, int hash_size)
{
    return ((k.pointsize^(3*hash_size+1))+(int)k.face)%hash_size;
}


#endif



/*
 *  HelpUI.h: Abstract interface to mosaic...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_HelpUI_h
#define SCI_project_HelpUI_h 1

class clString;

class HelpUI {
    HelpUI();
public:
    static void load(const clString& name);
};
#endif

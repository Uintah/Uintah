
/*
 *  MenuBar.h: Menubar helpers
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MenuBar_h
#define SCI_project_MenuBar_h 1

class CascadeButtonC;
class EncapsulatorC;
class PulldownMenuC;
class PushButtonC;
class RowColumnC;

class MenuC {
    CascadeButtonC* casc;
    PulldownMenuC* menu;
public:
    MenuC(EncapsulatorC& parent, char* name, int which);
    ~MenuC();
    RowColumnC* get_menu();
    PushButtonC* AddButton(char* name);
};

class MenuBarC {
    RowColumnC* rc;
    int nchild;
public:
    MenuBarC(EncapsulatorC& parent);
    ~MenuBarC();
    MenuC* AddMenu(char* name);
};

#endif

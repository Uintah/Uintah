
/*
 *  PopupMenu.h:
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

#include <Mt/RowColumn.h>

class PopupMenuC : public RowColumnC {
    Widget myparent;
public:
    PopupMenuC();
    virtual ~PopupMenuC();
    void Create( EncapsulatorC &parent,
		 const StringC& name );
protected:
    virtual Widget CreateWidget( Widget parent,
				 String name,
				 WidgetClass clas,
				 ArgList args,
				 Cardinal number );

};

#endif

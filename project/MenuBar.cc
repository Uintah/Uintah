
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

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <MenuBar.h>
#include <Mt/CascadeButton.h>
#include <Mt/PushButton.h>
#include <Mt/RowColumn.h>

class PulldownMenuC : public RowColumnC {
    Widget myparent;
public:
    PulldownMenuC();
    virtual ~PulldownMenuC();
    void Create( EncapsulatorC &parent,
		 const StringC& name );
protected:
    virtual Widget CreateWidget( Widget parent,
				 String name,
				 WidgetClass clas,
				 ArgList args,
				 Cardinal number );

};

PulldownMenuC::PulldownMenuC()
{
}

PulldownMenuC::~PulldownMenuC()
{
}

void PulldownMenuC::Create(EncapsulatorC& parent,
			   const StringC& name)
{
    myparent=parent;
    RowColumnC::Create(*(EncapsulatorC*)0, name);
}

Widget PulldownMenuC::CreateWidget( Widget,
				   String name,
				   WidgetClass,
				   ArgList args,
				   Cardinal number )
{
    return XmCreatePulldownMenu(myparent, name, args, number);
}

MenuBarC::MenuBarC(EncapsulatorC& parent)
: nchild(0)
{
    rc=new RowColumnC;
    rc->SetRowColumnType(XmMENU_BAR);
    rc->SetIsHomogeneous(True);
    rc->SetOrientation(XmHORIZONTAL);
    rc->SetEntryClass(xmCascadeButtonWidgetClass);
    rc->Create(parent, "menubar");
}

MenuBarC::~MenuBarC()
{
    delete rc;
}

MenuC* MenuBarC::AddMenu(char* name)
{
     return new MenuC(*rc, name, nchild++);
}

MenuC::MenuC(EncapsulatorC& parent, char* name, int which)
{
    menu=new PulldownMenuC;
    menu->Create(parent, name);
    casc=new CascadeButtonC;
    casc->SetSubMenuId(*menu);
    casc->Create(parent, name);
}

MenuC::~MenuC()
{
}

PushButtonC* MenuC::AddButton(char* name)
{
    PushButtonC* pb=new PushButtonC;
    pb->Create(*menu, name);
    return pb;
}

RowColumnC* MenuC::get_menu()
{
    return menu;
}


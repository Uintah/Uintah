
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

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <PopupMenu.h>

PopupMenuC::PopupMenuC()
{
}

PopupMenuC::~PopupMenuC()
{
}

void PopupMenuC::Create(EncapsulatorC& parent, const StringC& name)
{
    myparent=parent;
    RowColumnC::Create(*(EncapsulatorC*)0, name);
}

Widget PopupMenuC::CreateWidget(Widget parent,
				String name,
				WidgetClass,
				ArgList args,
				Cardinal number )
{
    return XmCreatePopupMenu(myparent, name, args, number);
}

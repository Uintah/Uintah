
/*
 *  Modules.cc: Build the database of Modules...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ModuleList.h>
#include <Classlib/String.h>
Module* make_WidgetReal();

void ModuleList::initialize_list()
{
    ModuleList::insert("WidgetReal", make_WidgetReal);
}

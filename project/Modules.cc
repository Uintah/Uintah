
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

void ModuleList::initialize_list()
{
    Module* make_WidgetReal();
    ModuleList::insert("WidgetReal", make_WidgetReal);
    Module* make_SoundFilter();
    ModuleList::insert("SoundFilter", make_SoundFilter);
    Module* make_SoundInput();
    ModuleList::insert("SoundInput", make_SoundInput);
    Module* make_SoundMixer();
    ModuleList::insert("SoundMixer", make_SoundMixer);
    Module* make_SoundOutput();
    ModuleList::insert("SoundOutput", make_SoundOutput);
}

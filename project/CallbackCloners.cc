
/*
 *  CallbackCloners.h: functions for cloning callback data
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

#include <CallbackCloners.h>
#include <NotFinished.h>
#include <Classlib/Assert.h>
#include <Mt/FileSelectionBox.h>
#include <Mt/List.h>
#include <Mt/Scale.h>

CallbackData* CallbackCloners::event_clone(void* event)
{
    return new CallbackData((XEvent*)event);
}

CallbackData* CallbackCloners::scale_clone(void* vdata)
{
    XmScaleCallbackStruct* cbs=(XmScaleCallbackStruct*)vdata;
    return new CallbackData(cbs->value);
}

CallbackData* CallbackCloners::selection_clone(void* vdata)
{
    XmFileSelectionBoxCallbackStruct* cbs=(XmFileSelectionBoxCallbackStruct*)vdata;
    char* str;
    if(!XmStringGetLtoR(cbs->value, (char*)XmSTRING_DEFAULT_CHARSET, &str))
	return new CallbackData("Internal Error");
    CallbackData* cbdata=new CallbackData(str);
    free(str);
    return cbdata;
}

CallbackData* CallbackCloners::list_clone(void* vdata)
{
    XmListCallbackStruct* cbs=(XmListCallbackStruct*)vdata;
    char* str;
    if(!XmStringGetLtoR(cbs->item, (char*)XmSTRING_DEFAULT_CHARSET, &str))
	return new CallbackData("Internal Error");
    CallbackData* cbdata=new CallbackData(str);
    free(str);
    return cbdata;
}

CallbackData::CallbackData(const clString& string_data)
: string_data(string_data), type(TypeString)
{
}

CallbackData::CallbackData(int int_data)
: int_data(int_data), type(TypeInt)
{
}

CallbackData::CallbackData(XEvent* event)
: event(*event), type(TypeEvent)
{
}

int CallbackData::get_int()
{
    ASSERT(type==TypeInt);
    return int_data;
}

clString CallbackData::get_string()
{
    ASSERT(type==TypeString);
    return string_data;
}

XEvent* CallbackData::get_event()
{
    ASSERT(type==TypeEvent);
    return &event;
}

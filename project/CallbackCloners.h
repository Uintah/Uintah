
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

#ifndef SCI_project_CallbackCloners_h
#define SCI_project_CallbackCloners_h 1

#include <Classlib/String.h>
#include <X11/Xlib.h>

class CallbackData {
    int int_data;
    clString string_data;
    enum Type {
	TypeInt, TypeString, TypeEvent,
    };
    Type type;
    XEvent event;
public:
    CallbackData(XEvent*);
    CallbackData(const clString&);
    CallbackData(int);
    int get_int();
    clString get_string();
    XEvent* get_event();
};

class CallbackCloners {
    CallbackCloners();
public:
    static CallbackData* input_clone(void*);
    static CallbackData* scale_clone(void*);
    static CallbackData* selection_clone(void*);
    static CallbackData* event_clone(void*);
    static CallbackData* list_clone(void*);
};

#endif


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

class CallbackData {
public:
    virtual ~CallbackData();
};

class CallbackCloners {
    CallbackCloners();
public:
    //static CallbackData* ???();
};

#endif

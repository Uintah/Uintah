
/*
 *  MessageTypes.h: enum definitions for Message Types...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MessageTypes_h
#define SCI_project_MessageTypes_h 1

class MessageTypes {
    MessageTypes();
public:
    enum MessageType {
	DoCallback,
	ExecuteModule,
	ReSchedule,

	MUIDispatch,
    };
};

#endif

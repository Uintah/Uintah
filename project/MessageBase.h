
/*
 *  XQColor.h: Automagically quantizing colors for X
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MessageBase_h
#define SCI_project_MessageBase_h 1

#include <MessageTypes.h>

class MessageBase {
public:
    MessageTypes::MessageType type;
    MessageBase(MessageTypes::MessageType type): type(type) {}
};

#endif

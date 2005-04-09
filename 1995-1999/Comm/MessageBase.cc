
/*
 *  MessageBase.cc: Base class for messages
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Comm/MessageBase.h>

MessageBase::MessageBase(MessageTypes::MessageType type)
: type(type)
{
}

MessageBase::~MessageBase()
{
}

//static char *id="@(#) $Id$";

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

namespace PSECommon {
namespace Comm {

MessageBase::MessageBase(MessageTypes::MessageType type)
: type(type)
{
}

MessageBase::~MessageBase()
{
}

} // End namespace Comm
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:44  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

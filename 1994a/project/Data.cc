
/*
 *  Data.cc: Base class for data items
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Data.h>

InData::InData(int protocol)
: protocol(protocol), connection(0)
{
}

int InData::using_protocol()
{
    return protocol;
}

OutData::OutData(int protocol)
: protocol(protocol), connection(0)
{
}

int OutData::using_protocol()
{
    return protocol;
}


/*
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 *  $Id$
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Exceptions/DimensionMismatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using SCICore::Exceptions::DimensionMismatch;


DimensionMismatch::DimensionMismatch(long value, long expected)
    : value(value), expected(expected)
{
    // Format the message now...
    char buf[120];
    sprintf(buf, "Dimension mismatch, got %ld, expeced %ld", value, expected);
    msg=strdup(buf);
}

DimensionMismatch::DimensionMismatch(const DimensionMismatch& copy)
    : msg(strdup(copy.msg))
{
}
    
DimensionMismatch::~DimensionMismatch()
{
    free(msg);
}

const char* DimensionMismatch::message() const
{
    return msg;
}

const char* DimensionMismatch::type() const
{
    return "SCICore::Exceptions::DimensionMismatch";
}

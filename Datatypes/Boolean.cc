
/*
 *  sciBoolean.h: All this for true and false...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Datatypes/Boolean.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew sciBoolean(0);
}

PersistentTypeID sciBoolean::type_id("Boolean", "Datatype", maker);

sciBoolean::sciBoolean(int value)
: value(value)
{
}

sciBoolean::sciBoolean(const sciBoolean& c)
: value(c.value)
{
}

sciBoolean::~sciBoolean()
{
}

sciBoolean* sciBoolean::clone() const
{
    return scinew sciBoolean(*this);
}

#define BOOLEAN_VERSION 1

void sciBoolean::io(Piostream& stream)
{
    stream.begin_class("Boolean", BOOLEAN_VERSION);
    stream.io(value);
    stream.end_class();
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<sciBoolean>;

#endif

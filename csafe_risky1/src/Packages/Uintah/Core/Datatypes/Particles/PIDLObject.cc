
/*
 *  PIDLObject:
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Uintah/Datatypes/Particles/PIDLObject.h>

using Uintah::Datatypes::PIDLObject;
using SCICore::PersistentSpace::PersistentTypeID;
PersistentTypeID PIDLObject::type_id("PIDLObject", "Datatype", 0);

PIDLObject::PIDLObject(const Component::PIDL::Object& obj)
    : obj(obj)
{
}

PIDLObject::~PIDLObject()
{
}

PIDLObject::PIDLObject(const PIDLObject& copy)
    : obj(copy.obj)
{
}

#define PIDLOBJECT_VERSION 1

void PIDLObject::io(Piostream& stream)
{
    stream.begin_class("PIDLObject", PIDLOBJECT_VERSION);
    stream.end_class();
}

//
// $Log$
// Revision 1.1  1999/10/07 02:08:23  sparker
// use standard iostreams and complex type
//
//




/*
 *  Class: Implementation of CIA.Class for PIDL
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Component/CIA/CIA_sidl.h>
#include <SCICore/Util/NotFinished.h>

using CIA::Class;
using CIA::Class_interface;
using CIA::Method;
using CIA::Object;

Object Class_interface::newInstance()
{
    NOT_FINISHED("final .CIA.Object .CIA.Class.newInstance()throws .CIA.InstantiationException");
    return 0;
}

bool Class_interface::isInterface()
{
    NOT_FINISHED("final bool .CIA.Class.isInterface()");
    return false;
}

bool Class_interface::isArray()
{
    NOT_FINISHED("final bool .CIA.Class.isArray()");
    return false;
}

bool Class_interface::isPrimitive()
{
    NOT_FINISHED("final bool .CIA.Class.isPrimitive()");
    return false;
}

Class Class_interface::getSuperclass()
{
    NOT_FINISHED("final .CIA.Class .CIA.Class.getSuperclass()");
    return 0;
}

Class Class_interface::getComponentType()
{
    NOT_FINISHED("final .CIA.Class .CIA.Class.getComponentType()");
    return 0;
}

Method Class_interface::getMethod()
{
    NOT_FINISHED("final .CIA.Method .CIA.Class.getMethod()throws .CIA.NoSuchMethodException");
    return 0;
}


//
// $Log$
// Revision 1.1  1999/09/24 06:26:22  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
//


/*
 *  Interface: Implementation of CIA.Interface for PIDL
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
using CIA::Interface;
using CIA::Interface_interface;
using CIA::Object;

Object Interface_interface::addReference()
{
    NOT_FINISHED(".CIA.Object .CIA.Interface.addReference()");
    return 0;
}

void Interface_interface::deleteReference()
{
    NOT_FINISHED("void .CIA.Interface.deleteReference()");
}

Class Interface_interface::getClass()
{
    NOT_FINISHED(".CIA.Class .CIA.Interface.getClass()");
    return 0;
}

bool Interface_interface::isSame(const Interface& object)
{
    NOT_FINISHED("bool .CIA.Interface.isSame(in .CIA.Interface object)");
    return false;
}

bool Interface_interface::isInstanceOf(const Class& type)
{
    NOT_FINISHED("bool .CIA.Interface.isInstanceOf(in .CIA.Class type)");
    return false;
}

bool Interface_interface::supportsInterface(const Class& type)
{
    NOT_FINISHED("bool .CIA.Interface.supportsInterface(in .CIA.Class type)");
    return false;
}

Interface Interface_interface::queryInterface(const Class& type)
{
    NOT_FINISHED(".CIA.Interface .CIA.Interface.queryInterface(in .CIA.Class type)");
    return false;
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

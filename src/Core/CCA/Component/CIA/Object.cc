
/*
 *  Object: Implementation of CIA.Object for PIDL
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

using CIA::Class;
using CIA::Interface;
using CIA::Object_interface;
using CIA::Object;

/*
 * These are all implemented in CIA.interface, so these are just
 * up calls, since they will get generated from cia spec.
 */

Object Object_interface::addReference()
{
    return Interface_interface::addReference();
}

void Object_interface::deleteReference()
{
    Interface_interface::deleteReference();
}

Class Object_interface::getClass()
{
    return Interface_interface::getClass();
}

bool Object_interface::isSame(const Interface& i)
{
    return Interface_interface::isSame(i);
}

bool Object_interface::isInstanceOf(const Class& c)
{
    return Interface_interface::isInstanceOf(c);
}

bool Object_interface::supportsInterface(const Class& c)
{
    return Interface_interface::supportsInterface(c);
}

Interface Object_interface::queryInterface(const Class& c)
{
    return Interface_interface::queryInterface(c);
}

//
// $Log$
// Revision 1.1  1999/09/24 06:26:23  sparker
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

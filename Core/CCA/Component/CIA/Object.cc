
/*
 *  Object: Implementation of CIA.Object for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/CIA/CIA_sidl.h>

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


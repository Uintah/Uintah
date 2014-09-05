
/*
 *  Method: Implementation of CIA.Method for PIDL
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
using CIA::Method_interface;

Class Method_interface::getDeclaringClass()
{
    NOT_FINISHED("Method_interface::getDeclaringClass");
    return 0;
}

::CIA::string Method_interface::getName()
{
    NOT_FINISHED("final string .CIA.Metho.getName()");
    return "";
}

Class Method_interface::getReturnType()
{
    NOT_FINISHED("Method_interface::getReturnType");
    return 0;
}

//
// $Log$
// Revision 1.2  1999/09/28 08:19:48  sparker
// Implemented start of array class (only 1d currently)
// Implement string class (typedef to std::string)
// Updates to spec now that sidl support strings
//
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

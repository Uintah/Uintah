
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ScalarFieldPort.h>

clString SimpleIPort<ScalarFieldHandle>::port_type("ScalarField");
clString SimpleIPort<ScalarFieldHandle>::port_color("VioletRed2");


#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ScalarFieldHandle>;
template class SimpleOPort<ScalarFieldHandle>;
template class SimplePortComm<ScalarFieldHandle>;
template class Mailbox<SimplePortComm<ScalarFieldHandle>*>;

#endif


/*
 *  MatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/MatrixPort.h>

clString SimpleIPort<MatrixHandle>::port_type("Matrix");
clString SimpleIPort<MatrixHandle>::port_color("dodger blue");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<MatrixHandle>;
template class SimpleOPort<MatrixHandle>;
template class SimplePortComm<MatrixHandle>;
template class Mailbox<SimplePortComm<MatrixHandle>*>;

#endif

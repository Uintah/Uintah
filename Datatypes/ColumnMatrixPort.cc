
/*
 *  ColumnMatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ColumnMatrixPort.h>

clString SimpleIPort<ColumnMatrixHandle>::port_type("ColumnMatrix");
clString SimpleIPort<ColumnMatrixHandle>::port_color("dodgerblue4");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ColumnMatrixHandle>;
template class SimpleOPort<ColumnMatrixHandle>;
template class SimplePortComm<ColumnMatrixHandle>;
template class Mailbox<SimplePortComm<ColumnMatrixHandle>*>;

#endif

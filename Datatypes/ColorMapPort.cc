
/*
 *  ColorMapPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ColorMapPort.h>

clString SimpleIPort<ColorMapHandle>::port_type("ColorMap");
clString SimpleIPort<ColorMapHandle>::port_color("blueviolet");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ColorMapHandle>;
template class SimpleOPort<ColorMapHandle>;
template class SimplePortComm<ColorMapHandle>;
template class Mailbox<SimplePortComm<ColorMapHandle>*>;

#endif

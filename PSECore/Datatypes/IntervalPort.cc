//static char *id="@(#) $Id$";

/*
 *  IntervalPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/IntervalPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<IntervalHandle>::port_type("Interval");
clString SimpleIPort<IntervalHandle>::port_color("mediumseagreen");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

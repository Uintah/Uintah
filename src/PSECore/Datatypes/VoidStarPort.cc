//static char *id="@(#) $Id$";

/*
 *  VoidStarPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <CommonDatatypes/VoidStarPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<VoidStarHandle>::port_type("VoidStar");
clString SimpleIPort<VoidStarHandle>::port_color("gold1");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:51  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

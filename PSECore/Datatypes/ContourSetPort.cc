//static char *id="@(#) $Id$";

/*
 *  ContourSetPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/ContourSetPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<ContourSetHandle>::port_type("ContourSet");
clString SimpleIPort<ContourSetHandle>::port_color("#388e8e");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//

//static char *id="@(#) $Id$";

/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/VectorFieldPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<VectorFieldHandle>::port_type("VectorField");
clString SimpleIPort<VectorFieldHandle>::port_color("orchid4");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

//static char *id="@(#) $Id$";

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

#include <CommonDatatypes/ScalarFieldPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<ScalarFieldHandle>::port_type("ScalarField");
clString SimpleIPort<ScalarFieldHandle>::port_color("VioletRed2");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:49  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

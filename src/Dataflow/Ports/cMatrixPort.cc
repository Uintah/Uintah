//static char *id="@(#) $Id$";

/*
 *  cMatrixPort.h
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <CommonDatatypes/cMatrixPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<cMatrixHandle>::port_type("cMatrix");
clString SimpleIPort<cMatrixHandle>::port_color("red");

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

//static char *id="@(#) $Id$";

/*
 *  VizGridPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Particles/VizGridPort.h>

//namespace Uintah {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace Uintah::Datatypes;

clString SimpleIPort<VizGridHandle>::port_type("VizGrid");
clString SimpleIPort<VizGridHandle>::port_color("gray50");

//} // End namespace Datatypes
//} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/09/21 16:08:31  kuzimmer
// modifications for binary file format
//
// Revision 1.1  1999/07/27 16:59:00  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

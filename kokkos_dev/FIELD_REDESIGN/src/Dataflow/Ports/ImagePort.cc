//static char *id="@(#) $Id$";

/*
 *  ImagePort.cc
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCIRun/Datatypes/Image/ImagePort.h>

using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;

template<> clString SimpleIPort<ImageHandle>::port_type("Image");
template<> clString SimpleIPort<ImageHandle>::port_color("misty rose");


//
// $Log$
// Revision 1.2  1999/08/31 08:55:29  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.1  1999/07/27 16:58:46  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 22:25:36  dav
// trying to update all
//
// Revision 1.1  1999/04/29 21:50:58  dav
// moved ImagePort datatype out of common and into scirun
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

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

#include <Datatypes/Image/ImagePort.h>

//namespace SCIRun {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace SCIRun::Datatypes;

clString SimpleIPort<ImageHandle>::port_type("Image");
clString SimpleIPort<ImageHandle>::port_color("misty rose");

//} // End namespace Datatypes
//} // End namespace SCIRun

//
// $Log$
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

//static char *id="@(#) $Id$";

/*
 *  SegFld.cc: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/Datatypes/General/SegFldPort.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

template<> clString SimpleIPort<SegFldHandle>::port_type("SegFld");
template<> clString SimpleIPort<SegFldHandle>::port_color("Green");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/30 20:19:20  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.1  1999/08/23 02:53:00  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:05  dmw
// Added and updated DaveW Datatypes/Modules
//
//

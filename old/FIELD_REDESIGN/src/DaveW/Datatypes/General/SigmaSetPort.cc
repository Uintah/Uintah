//static char *id="@(#) $Id$";

/*
 *  SigmaSetPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <DaveW/Datatypes/General/SigmaSetPort.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

template<> clString SimpleIPort<SigmaSetHandle>::port_type("SigmaSet");
template<> clString SimpleIPort<SigmaSetHandle>::port_color("chocolate4");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/30 20:19:20  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.1  1999/08/23 02:53:01  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:07  dmw
// Added and updated DaveW Datatypes/Modules
//
//

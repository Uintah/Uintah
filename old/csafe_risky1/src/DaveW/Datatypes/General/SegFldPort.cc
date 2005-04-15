//static char *id="@(#) $Id$";

/*
 *  SegFldPort.cc: The SegFldPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
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
// Revision 1.3  1999/09/01 05:27:36  dmw
// more DaveW datatypes...
//
//

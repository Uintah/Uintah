//static char *id="@(#) $Id$";

/*
 *  ContourSetPort.cc: The ContourSetPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/ContourSetPort.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

template<> clString SimpleIPort<ContourSetHandle>::port_type("ContourSet");
template<> clString SimpleIPort<ContourSetHandle>::port_color("#388e8e");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 05:27:35  dmw
// more DaveW datatypes...
//
//

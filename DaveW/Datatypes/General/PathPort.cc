//static char *id="@(#) $Id$";

/*
 *  PathPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/PathPort.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

template<> clString SimpleIPort<PathHandle>::port_type("Path");
template<> clString SimpleIPort<PathHandle>::port_color("chocolate4");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//

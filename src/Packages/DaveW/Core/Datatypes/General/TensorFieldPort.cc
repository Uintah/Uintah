//static char *id="@(#) $Id$";

/*
 *  TensorFieldPort.cc: The TensorFieldPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/TensorFieldPort.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

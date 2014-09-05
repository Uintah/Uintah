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
#include <DaveW/share/share.h>
#include <SCICore/Malloc/Allocator.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

extern "C" {
DaveWSHARE IPort* make_ContourSetIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<ContourSetHandle>(module,name);
}
DaveWSHARE OPort* make_ContourSetOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<ContourSetHandle>(module,name);
}
}

template<> clString SimpleIPort<ContourSetHandle>::port_type("ContourSet");
template<> clString SimpleIPort<ContourSetHandle>::port_color("#388e8e");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.3  2000/11/29 09:49:30  moulding
// changed all instances of "new" to "scinew"
//
// Revision 1.2  2000/11/22 17:30:15  moulding
// added extern "C" make functions for input and output ports (to be used
// by the autoport facility).
//
// Revision 1.1  1999/09/01 05:27:35  dmw
// more DaveW datatypes...
//
//

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
#include <DaveW/share/share.h>
#include <SCICore/Malloc/Allocator.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

extern "C" {
DaveWSHARE IPort* make_SegFldIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<SegFldHandle>(module,name);
}
DaveWSHARE OPort* make_SegFldOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<SegFldHandle>(module,name);
}
}

template<> clString SimpleIPort<SegFldHandle>::port_type("SegFld");
template<> clString SimpleIPort<SegFldHandle>::port_color("Green");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.5  2000/11/29 09:49:30  moulding
// changed all instances of "new" to "scinew"
//
// Revision 1.4  2000/11/22 17:30:15  moulding
// added extern "C" make functions for input and output ports (to be used
// by the autoport facility).
//
// Revision 1.3  1999/09/01 05:27:36  dmw
// more DaveW datatypes...
//
//

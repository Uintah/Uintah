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
#include <DaveW/share/share.h>
#include <SCICore/Malloc/Allocator.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

extern "C" {
DaveWSHARE IPort* make_SigmaSetIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<SigmaSetHandle>(module,name);
}
DaveWSHARE OPort* make_SigmaSetOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<SigmaSetHandle>(module,name);
}
}

template<> clString SimpleIPort<SigmaSetHandle>::port_type("SigmaSet");
template<> clString SimpleIPort<SigmaSetHandle>::port_color("chocolate4");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.4  2000/11/29 09:49:30  moulding
// changed all instances of "new" to "scinew"
//
// Revision 1.3  2000/11/22 17:30:16  moulding
// added extern "C" make functions for input and output ports (to be used
// by the autoport facility).
//
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

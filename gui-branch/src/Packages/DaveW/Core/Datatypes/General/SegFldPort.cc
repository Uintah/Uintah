
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

#include <Packages/DaveW/Core/Datatypes/General/SegFldPort.h>
#include <Packages/DaveW/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/DaveW {
//namespace Datatypes {

using namespace SCIRun;
using namespace DaveW;

extern "C" {
DAVEWSHARE IPort* make_SegFldIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<SegFldHandle>(module,name);
}
DAVEWSHARE OPort* make_SegFldOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<SegFldHandle>(module,name);
}
}

template<> clString SimpleIPort<SegFldHandle>::port_type("SegFld");
template<> clString SimpleIPort<SegFldHandle>::port_color("Green");

//} // End namespace Datatypes
//} // End namespace Packages/DaveW



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

#include <Packages/DaveW/Core/Datatypes/General/SigmaSetPort.h>
#include <Packages/DaveW/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/DaveW {
//namespace Datatypes {

using namespace SCIRun;
using namespace DaveW;

extern "C" {
DAVEWSHARE IPort* make_SigmaSetIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<SigmaSetHandle>(module,name);
}
DAVEWSHARE OPort* make_SigmaSetOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<SigmaSetHandle>(module,name);
}
}

template<> clString SimpleIPort<SigmaSetHandle>::port_type("SigmaSet");
template<> clString SimpleIPort<SigmaSetHandle>::port_color("chocolate4");

//} // End namespace Datatypes
//} // End namespace Packages/DaveW


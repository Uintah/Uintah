
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

#include <Packages/DaveW/Core/Datatypes/General/ContourSetPort.h>
#include <Packages/DaveW/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/DaveW {
//namespace Datatypes {

using namespace SCIRun;
using namespace DaveW;

extern "C" {
DAVEWSHARE IPort* make_ContourSetIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<ContourSetHandle>(module,name);
}
DAVEWSHARE OPort* make_ContourSetOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<ContourSetHandle>(module,name);
}
}

template<> clString SimpleIPort<ContourSetHandle>::port_type("ContourSet");
template<> clString SimpleIPort<ContourSetHandle>::port_color("#388e8e");

//} // End namespace Datatypes
//} // End namespace Packages/DaveW



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

#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Packages/DaveW/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/DaveW {
//namespace Datatypes {

using namespace SCIRun;
using namespace DaveW;

extern "C" {
DAVEWSHARE IPort* make_TensorFieldIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<TensorFieldHandle>(module,name);
}
DAVEWSHARE OPort* make_TensorFieldOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<TensorFieldHandle>(module,name);
}
}

template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");

//} // End namespace Datatypes
//} // End namespace Packages/DaveW


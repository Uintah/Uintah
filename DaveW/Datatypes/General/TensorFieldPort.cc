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
#include <DaveW/share/share.h>

//namespace DaveW {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace DaveW::Datatypes;

extern "C" {
DaveWSHARE IPort* make_TensorFieldIPort(Module* module,
					 const clString& name) {
  return new SimpleIPort<TensorFieldHandle>(module,name);
}
DaveWSHARE OPort* make_TensorFieldOPort(Module* module,
					 const clString& name) {
  return new SimpleOPort<TensorFieldHandle>(module,name);
}
}

template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");

//} // End namespace Datatypes
//} // End namespace DaveW

//
// $Log$
// Revision 1.2  2000/11/22 17:30:16  moulding
// added extern "C" make functions for input and output ports (to be used
// by the autoport facility).
//
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

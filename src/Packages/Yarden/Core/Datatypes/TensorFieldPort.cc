/*
 *  TensorFieldPort.cc: The TensorFieldPort datatype
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Yarden/Datatypes/TensorFieldPort.h>
#include <Yarden/share/share.h>

namespace SCICore {
  namespace Datatypes {
    
    using namespace SCICore::Containers;
    using namespace DaveW::Datatypes;
    
    extern "C" {
      YardenSHARE IPort* make_MatrixIPort(Module* module,
					   const clString& name) {
	return new SimpleIPort<MatrixHandle>(module,name);
      }
      YardenSHARE OPort* make_MatrixOPort(Module* module,
					   const clString& name) {
	return new SimpleOPort<MatrixHandle>(module,name);
      }
    }

    template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
    template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");
  }
}

//
// $Log$
// Revision 1.2  2000/11/22 18:53:46  moulding
// added extern "C" make functions for input and output ports (to be used
// by the autoport facility).
//
// Revision 1.1  2000/10/23 23:39:39  yarden
// Tensor and Tensor Field definitions
//
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

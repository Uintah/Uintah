/*
 *  TensorFieldPort.cc: The TensorFieldPort datatype
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Yarden/Core/Datatypes/TensorFieldPort.h>
#include <Packages/Yarden/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace Yarden {
using namespace SCIRun;
    using namespace DaveW::Datatypes;
    
    extern "C" {
      Packages/YardenSHARE IPort* make_MatrixIPort(Module* module,
					   const clString& name) {
	return scinew SimpleIPort<MatrixHandle>(module,name);
      }
      Packages/YardenSHARE OPort* make_MatrixOPort(Module* module,
					   const clString& name) {
	return scinew SimpleOPort<MatrixHandle>(module,name);
      }
    }

    template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
} // End namespace Yarden
    template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");


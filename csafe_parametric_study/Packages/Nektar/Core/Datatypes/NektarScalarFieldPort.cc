
/*
 *  Packages/NektarScalarFieldPort.cc: The Scalar Field Data type
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Nektar/Core/Datatypes/NektarScalarFieldPort.h>
#include <Packages/Nektar/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/Nektar {
//namespace Datatypes {

using namespace SCIRun;
using namespace Nektar::Datatypes;

extern "C" {
  Packages/NektarSHARE IPort* make_Packages/NektarScalarFieldIPort(Module* module,
						 const clString& name) {
    return scinew SimpleIPort<Packages/NektarScalarFieldHandle>(module,name);
  }
  Packages/NektarSHARE OPort* make_Packages/NektarScalarFieldOPort(Module* module, const 
						 clString& name) {
    return scinew SimpleOPort<Packages/NektarScalarFieldHandle>(module,name);
  }
}

template<> clString SimpleIPort<Packages/NektarScalarFieldHandle>::port_type("Packages/NektarScalarField");
template<> clString SimpleIPort<Packages/NektarScalarFieldHandle>::port_color("Blue");

//} // End namespace Datatypes
//} // End namespace Packages/Nektar


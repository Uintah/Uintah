
/*
 *  NektarScalarFieldPort.cc: The Scalar Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarScalarFieldPort.h>
#include <Nektar/share/share.h>
#include <SCICore/Malloc/Allocator.h>

//namespace Nektar {
//namespace Datatypes {

using namespace SCICore::Containers;
using namespace Nektar::Datatypes;

extern "C" {
  NektarSHARE IPort* make_NektarScalarFieldIPort(Module* module,
						 const clString& name) {
    return scinew SimpleIPort<NektarScalarFieldHandle>(module,name);
  }
  NektarSHARE OPort* make_NektarScalarFieldOPort(Module* module, const 
						 clString& name) {
    return scinew SimpleOPort<NektarScalarFieldHandle>(module,name);
  }
}

template<> clString SimpleIPort<NektarScalarFieldHandle>::port_type("NektarScalarField");
template<> clString SimpleIPort<NektarScalarFieldHandle>::port_color("Blue");

//} // End namespace Datatypes
//} // End namespace Nektar


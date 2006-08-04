
/*
 *  Packages/NektarVectorField.cc: The Vector Field Data type
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Nektar/Core/Datatypes/NektarVectorFieldPort.h>
#include <Packages/Nektar/share/share.h>
#include <Core/Malloc/Allocator.h>

//namespace Packages/Nektar {
//namespace Datatypes {
    
using namespace SCIRun;
using namespace Nektar::Datatypes;

extern "C" {
  Packages/NektarSHARE IPort* make_Packages/NektarVectorFieldIPort(Module* module,
						 const clString& name) {
    return scinew SimpleIPort<Packages/NektarVectorFieldHandle>(module,name);
  }
  Packages/NektarSHARE OPort* make_Packages/NektarVectorFieldOPort(Module* module, const 
						 clString& name) {
    return scinew SimpleOPort<Packages/NektarVectorFieldHandle>(module,name);
  }
}

template<> clString SimpleIPort<Packages/NektarVectorFieldHandle>::port_type("Packages/NektarVectorField");
template<> clString SimpleIPort<Packages/NektarVectorFieldHandle>::port_color("LightBlue");

//  } // End namespace Datatypes
//} // End namespace Packages/Nektar



/*
 *  NektarVectorField.cc: The Vector Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarVectorFieldPort.h>
#include <Nektar/share/share.h>

//namespace Nektar {
//namespace Datatypes {
    
using namespace SCICore::Containers;
using namespace Nektar::Datatypes;

extern "C" {
  NektarSHARE IPort* make_NektarVectorFieldIPort(Module* module,
						 const clString& name) {
    return new SimpleIPort<NektarVectorFieldHandle>(module,name);
  }
  NektarSHARE OPort* make_NektarVectorFieldOPort(Module* module, const 
						 clString& name) {
    return new SimpleOPort<NektarVectorFieldHandle>(module,name);
  }
}

template<> clString SimpleIPort<NektarVectorFieldHandle>::port_type("NektarVectorField");
template<> clString SimpleIPort<NektarVectorFieldHandle>::port_color("LightBlue");

//  } // End namespace Datatypes
//} // End namespace Nektar


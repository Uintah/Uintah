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

namespace SCICore {
  namespace Datatypes {
    
    using namespace SCICore::Containers;
    using namespace DaveW::Datatypes;
    
    template<> clString SimpleIPort<TensorFieldHandle>::port_type("TensorField");
    template<> clString SimpleIPort<TensorFieldHandle>::port_color("green3");
  }
}

//
// $Log$
// Revision 1.1.2.1  2000/10/26 10:06:45  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.1  2000/10/23 23:39:39  yarden
// Tensor and Tensor Field definitions
//
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

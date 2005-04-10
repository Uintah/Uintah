
/*
 *  TensorFieldPort.h: The TensorFieldPort datatype
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_Datatypes_TensorFieldPort_h
#define SCI_Datatypes_TensorFieldPort_h 

#include <PSECore/Datatypes/SimplePort.h>
#include <Yarden/Datatypes/TensorField.h>

namespace SCICore {
  namespace Datatypes {
    
    using namespace PSECore::Datatypes;
    
    typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
    typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;
    
  } // End namespace Datatypes
} // End namespace Yarden

//
// $Log$
// Revision 1.1  2000/10/23 23:39:39  yarden
// Tensor and Tensor Field definitions
//
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

#endif

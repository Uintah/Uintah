
/*
 *  TensorFieldPort.h: The TensorFieldPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_TensorFieldPort_h
#define SCI_DaveW_Datatypes_TensorFieldPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <DaveW/Datatypes/General/TensorFieldBase.h>

namespace DaveW {
namespace Datatypes {

using namespace PSECore::Datatypes;

typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//
//

#endif

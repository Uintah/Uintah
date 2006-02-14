
/*
 *  ContourSetPort.h: The ContourSetPort datatype
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_ContourSetPort_h
#define SCI_Packages_DaveW_Datatypes_ContourSetPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/DaveW/Core/Datatypes/General/ContourSet.h>

namespace DaveW {
using namespace SCIRun;

typedef SimpleIPort<ContourSetHandle> ContourSetIPort;
typedef SimpleOPort<ContourSetHandle> ContourSetOPort;
} // End namespace DaveW



#endif

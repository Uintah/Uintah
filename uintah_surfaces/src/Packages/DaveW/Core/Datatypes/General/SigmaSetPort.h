
/*
 *  SigmaSetPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_SigmaSetPort_h
#define SCI_Packages_DaveW_Datatypes_SigmaSetPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/DaveW/Core/Datatypes/General/SigmaSet.h>

namespace DaveW {
using namespace SCIRun;

typedef SimpleIPort<SigmaSetHandle> SigmaSetIPort;
typedef SimpleOPort<SigmaSetHandle> SigmaSetOPort;
} // End namespace DaveW



#endif

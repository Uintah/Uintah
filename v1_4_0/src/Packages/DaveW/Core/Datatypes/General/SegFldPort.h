
/*
 *  SegFld.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_SegFldPort_h
#define SCI_Packages_DaveW_Datatypes_SegFldPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/DaveW/Core/Datatypes/General/SegFld.h>

namespace DaveW {
using namespace SCIRun;

typedef SimpleIPort<SegFldHandle> SegFldIPort;
typedef SimpleOPort<SegFldHandle> SegFldOPort;
} // End namespace DaveW



#endif

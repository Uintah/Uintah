/*
 *  CameraView.h
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_Dataflow_Datatypes_CameraViewPort_h
#define SCI_Dataflow_Datatypes_CameraViewPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/CameraView.h>

namespace SCIRun {

    
    typedef SimpleIPort<CameraViewHandle> CameraViewIPort;
    typedef SimpleOPort<CameraViewHandle> CameraViewOPort;

} // End namespace SCIRun

#endif

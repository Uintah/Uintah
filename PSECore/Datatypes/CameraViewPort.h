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

#ifndef SCI_PSECore_Datatypes_CameraViewPort_h
#define SCI_PSECore_Datatypes_CameraViewPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/CameraView.h>

namespace PSECore {
  namespace Datatypes {

    using SCICore::Datatypes::CameraViewHandle;
    
    typedef SimpleIPort<CameraViewHandle> CameraViewIPort;
    typedef SimpleOPort<CameraViewHandle> CameraViewOPort;

  } // End namespace Datatypes
} // End namespace PSECore

#endif
